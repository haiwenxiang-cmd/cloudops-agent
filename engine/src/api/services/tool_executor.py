import asyncio
import json
import logging
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional

from ..config import settings
from ..integrations.jenkins import (
    filter_jenkins_tools,
    inject_jenkins_metadata_tool_args,
)
from ..models.mutation import ExecutionStatus, VerificationStatus
from ..utils.clock import now_ms
from ..utils.sanitization import mcp_tools_to_openai_format
from .approvals import ApprovalService
from .integrations import IntegrationService
from .mcp_client import MCPClient
from .mutation_journal import MutationJournalService
from .mutation_policy import MutationPolicyService
from .mutation_utils import attach_operation_metadata, hmac_fingerprint, redact_tool_args
from .mutation_verifier import CONTROLLED_MUTATION_TOOLS, build_verification_plan
from .tool_capabilities import (
    capability_error,
    get_tool_capability,
    is_effectively_read_only,
    is_infrastructure_tool,
)
from .tools_cache import ToolsCache

logger = logging.getLogger(__name__)

AVAILABLE_TOOLSETS = ("k8s", "helm", "argo", "jenkins", "memory")

LOAD_TOOLSET_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "load_toolset",
        "description": (
            "Load additional tool categories into the current session. "
            "By default only read-only Kubernetes tools are available. "
            "Call this to load Helm, Argo Rollouts, or Jenkins tools, "
            "or to enable write/mutation operations for any toolset. "
            "Newly loaded tools are not callable in the same response as "
            "load_toolset. They become available on the next model turn."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "toolset": {
                    "type": "string",
                    "enum": list(AVAILABLE_TOOLSETS),
                    "description": "The toolset category to load.",
                },
                "include_write_tools": {
                    "type": "boolean",
                    "description": (
                        "Set to true to include write/mutation tools "
                        "(apply, delete, scale, patch, install, etc.). "
                        "Leave false for read-only operations."
                    ),
                },
            },
            "required": ["toolset", "include_write_tools"],
        },
    },
}


def _resolve_tool_tag(tool: Dict[str, Any]) -> Optional[str]:
    tags = tool.get("tags")
    if isinstance(tags, list) and tags:
        return tags[0]

    meta = tool.get("meta") or {}
    fastmcp = meta.get("_fastmcp") or {}
    fm_tags = fastmcp.get("tags")
    if isinstance(fm_tags, list) and fm_tags:
        return fm_tags[0]

    name = tool.get("name", "")
    if isinstance(name, str):
        if name.startswith("k8s_") or name == "wait_for_x_seconds":
            return "k8s"
        if name.startswith("helm_"):
            return "helm"
        if name.startswith("argo_"):
            return "argo"
        if name.startswith("jenkins_"):
            return "jenkins"
        if name.startswith("memory_"):
            return "memory"

    return None


def _is_read_only(tool: Dict[str, Any]) -> bool:
    return is_effectively_read_only(str(tool.get("name") or ""), tool)


ALLOWED_TOOL_TAGS = frozenset(AVAILABLE_TOOLSETS)

_MEMORY_INTERNAL_PARAMS = frozenset({"_user_id", "_conversation_id", "_run_id"})


def _strip_internal_memory_params(tools: List[Dict[str, Any]]) -> None:
    """Hide all engine-owned parameters from the LLM-visible tool schemas."""
    for tool in tools:
        fn = tool.get("function", {})
        if not isinstance(fn.get("name"), str):
            continue
        params = fn.get("parameters", {})
        props = params.get("properties")
        if isinstance(props, dict):
            if fn["name"].startswith("memory_"):
                for key in _MEMORY_INTERNAL_PARAMS:
                    props.pop(key, None)
            for key in list(props):
                if str(key).startswith("_skyflo_") or key == "skyflo_operation_id":
                    props.pop(key, None)
        required = params.get("required")
        if isinstance(required, list):
            params["required"] = [
                r
                for r in required
                if (
                    (not fn["name"].startswith("memory_") or r not in _MEMORY_INTERNAL_PARAMS)
                    and not str(r).startswith("_skyflo_")
                    and r != "skyflo_operation_id"
                )
            ]


def filter_tools_by_loaded_toolsets(
    tools: List[Dict[str, Any]],
    loaded_toolsets: Dict[str, bool],
) -> List[Dict[str, Any]]:
    filtered = []
    for tool in tools:
        name = str(tool.get("name") or "")
        if capability_error(name, tool):
            continue
        tag = _resolve_tool_tag(tool)
        if tag is None or tag not in ALLOWED_TOOL_TAGS:
            continue

        if tag not in loaded_toolsets:
            continue

        include_write = loaded_toolsets[tag]
        if include_write or _is_read_only(tool):
            filtered.append(tool)

    return filtered


ProgressCallback = Callable[[str, Optional[float]], Awaitable[None]]
EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]

_AMBIGUOUS_ERROR_TYPES = frozenset(
    {
        "timeout",
        "transient",
        "transport_error",
        "connection_error",
        "cancelled",
        "mcp_protocol_error",
    }
)


def _mutation_meta_block(
    operation: Any,
    *,
    requires_verification: bool,
    title: Optional[str] = None,
    display_result: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    block = {
        "type": "skyflo.mutation",
        "operation_id": str(operation.id),
        "call_id": operation.call_id,
        "tool": operation.tool_name,
        "execution_status": operation.execution_status.value,
        "verification_status": operation.verification_status.value,
        "requires_verification": requires_verification,
    }
    if title:
        block["title"] = title
    if display_result is not None:
        block["display_result"] = display_result
    return block


def _reliability_dict(result: Dict[str, Any]) -> Dict[str, Any]:
    reliability = result.get("reliability") or {}
    if isinstance(reliability, list):
        reliability = reliability[0] if reliability else {}
    return reliability if isinstance(reliability, dict) else {}


class ToolExecutor:
    def __init__(
        self,
        approvals: Optional[ApprovalService] = None,
        sse_publish: Optional[EventCallback] = None,
        mcp_client: Optional[MCPClient] = None,
        owns_client: bool = True,
        tools_cache: Optional[ToolsCache] = None,
    ):
        self.mcp_url = settings.MCP_SERVER_URL
        self.sse_publish = sse_publish
        self._mcp_client: Optional[MCPClient] = mcp_client
        self._owns_client: bool = owns_client if mcp_client is None else False

        self._tools = tools_cache or ToolsCache()
        self._integrations = (
            IntegrationService(mcp_client=self._mcp_client) if mcp_client else IntegrationService()
        )
        self.mutation_policy = MutationPolicyService()
        self.mutation_journal = MutationJournalService()

        if approvals:
            self.approvals = approvals
            self.approvals.tool_metadata_fetcher = self._get_tool_metadata
        else:
            self.approvals = ApprovalService(tool_metadata_fetcher=self._get_tool_metadata)

    async def _get_mcp_client(self) -> MCPClient:
        if self._mcp_client is None:
            self._mcp_client = MCPClient()
            self._owns_client = True
        return self._mcp_client

    def invalidate_tools_cache(self) -> None:
        self._tools.invalidate()

    async def _fetch_tools_from_server(self) -> List[Any]:
        client = await self._get_mcp_client()
        return await client.list_tools_raw()

    async def _get_tool_metadata(self, tool_name: str) -> Optional[Dict[str, Any]]:
        try:
            return await self._tools.get_by_name(tool_name, self._fetch_tools_from_server)
        except Exception as e:
            logger.error(f"Error fetching metadata for tool '{tool_name}': {e}")
            return None

    @staticmethod
    def _is_system_tool(tool_metadata: Optional[Dict[str, Any]]) -> bool:
        if not tool_metadata:
            return False
        annotations = tool_metadata.get("annotations") or {}
        return bool(annotations.get("systemTool", False))

    async def close(self) -> None:
        self._mcp_client = None
        await self.approvals.close()

    async def filter_integrations_tools(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        try:
            jenkins_integration = await self._integrations.get_integration("jenkins")
            jenkins_configured = jenkins_integration is not None
            jenkins_status = jenkins_integration.status if jenkins_integration else None

            tools = filter_jenkins_tools(
                tools=tools,
                integration_status=jenkins_status,
                is_configured=jenkins_configured,
            )

            return tools
        except Exception as e:
            logger.error(f"Error filtering integration tools: {e}")
            return tools

    async def inject_integration_tool_params(
        self,
        tool_name: str,
        args: Dict[str, Any],
        tool_metadata: Optional[Dict[str, Any]],
        call_id: str,
        tool_title: str,
        run_id: str,
    ) -> tuple[Dict[str, Any], Optional[List[Dict[str, Any]]]]:
        jenkins_integration = await self._integrations.get_integration("jenkins")
        args, jenkins_error = inject_jenkins_metadata_tool_args(
            tool_name=tool_name,
            args=args,
            tool_metadata=tool_metadata,
            integration=jenkins_integration,
        )

        if jenkins_error:
            if self.sse_publish:
                await self.sse_publish(
                    {
                        "type": "tool.error",
                        "call_id": call_id,
                        "tool": tool_name,
                        "title": tool_title,
                        "error": jenkins_error,
                        "run_id": run_id,
                        "timestamp": now_ms(),
                    }
                )
            return args, [{"type": "text", "text": jenkins_error}]

        return args, None

    async def get_llm_compatible_tools(
        self, loaded_toolsets: Optional[Dict[str, bool]] = None
    ) -> List[Dict[str, Any]]:
        try:
            all_tools = await self._tools.get_all(self._fetch_tools_from_server)

            all_tools = await self.filter_integrations_tools(all_tools)

            if loaded_toolsets is not None:
                all_tools = filter_tools_by_loaded_toolsets(all_tools, loaded_toolsets)

            openai_tools = mcp_tools_to_openai_format({"tools": all_tools})
            openai_tools.append(LOAD_TOOLSET_TOOL)
            _strip_internal_memory_params(openai_tools)

            logger.debug(f"Tools provided: {len(openai_tools)} (toolsets={loaded_toolsets})")

            return openai_tools
        except Exception as e:
            logger.error(f"Error preparing OpenAI-compatible tools: {e}")
            try:
                client = await self._get_mcp_client()
                tools_raw = await client.get_tools()

                raw_list = (
                    tools_raw.get("tools", [])
                    if isinstance(tools_raw, dict)
                    else tools_raw
                    if isinstance(tools_raw, list)
                    else []
                )
                if loaded_toolsets is not None:
                    raw_list = filter_tools_by_loaded_toolsets(raw_list, loaded_toolsets)
                openai_tools = mcp_tools_to_openai_format({"tools": raw_list})
                openai_tools.append(LOAD_TOOLSET_TOOL)
                _strip_internal_memory_params(openai_tools)
                return openai_tools
            except Exception as inner:
                logger.error(f"Fallback tools fetch failed: {inner}")
                return [LOAD_TOOLSET_TOOL]

    class ApprovalPending(Exception):
        def __init__(self, call_id: str, tool: str):
            super().__init__(f"Approval pending for {tool} ({call_id})")
            self.call_id = call_id
            self.tool = tool

    async def execute(
        self,
        run_id: str,
        name: str,
        args: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        call_id: Optional[str] = None,
        operation_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        call_id = (call_id or str(uuid.uuid4())).strip()
        operation_id = (operation_id or str(uuid.uuid4())).strip()
        context = context or {}
        controlled_operation = None

        try:
            tool_metadata = await self._get_tool_metadata(name)
            tool_title = tool_metadata.get("title", name) if tool_metadata else name
            system_tool = self._is_system_tool(tool_metadata)
            publish_events = self.sse_publish is not None and not system_tool

            registry_error = capability_error(name, tool_metadata)
            if registry_error:
                if publish_events:
                    await self.sse_publish(
                        {
                            "type": "tool.error",
                            "call_id": call_id,
                            "tool": name,
                            "title": tool_title,
                            "error": registry_error,
                            "run_id": run_id,
                            "timestamp": now_ms(),
                        }
                    )
                return [{"type": "text", "text": registry_error}]

            args, integration_error = await self.inject_integration_tool_params(
                tool_name=name,
                args=args,
                tool_metadata=tool_metadata,
                call_id=call_id,
                tool_title=tool_title,
                run_id=run_id,
            )

            if integration_error is not None:
                return integration_error

            validation_error = await self._validate_tool_parameters(name, args, tool_metadata)
            if validation_error:
                return [
                    {
                        "type": "text",
                        "text": f"Tool validation failed: {validation_error}",
                    }
                ]

            event_args = redact_tool_args(name, args)

            capability = get_tool_capability(name)
            needs_approval = (
                capability.requires_approval
                if capability is not None and is_infrastructure_tool(name)
                else await self.approvals.need_approval(name, args)
            )
            infrastructure_mutation = bool(
                settings.MUTATION_CONTROL_ENABLED
                and needs_approval
                and is_infrastructure_tool(name)
            )
            controlled_mutation = bool(
                infrastructure_mutation
                and capability is not None
                and capability.controlled
                and name in CONTROLLED_MUTATION_TOOLS
            )

            policy_decision = None
            verification_plan = None
            if infrastructure_mutation:
                policy_decision = self.mutation_policy.evaluate(
                    tool_name=name,
                    args=args,
                    user_id=context.get("user_id"),
                    user_role=context.get("user_role"),
                    environment=context.get("environment"),
                )
                if not policy_decision.allowed:
                    if publish_events:
                        await self.sse_publish(
                            {
                                "type": "mutation.policy_denied",
                                "run_id": run_id,
                                "call_id": call_id,
                                "operation_id": operation_id,
                                "tool": name,
                                "rule": policy_decision.rule,
                                "reason": policy_decision.reason,
                                "timestamp": now_ms(),
                            }
                        )
                        await self.sse_publish(
                            {
                                "type": "tool.error",
                                "run_id": run_id,
                                "call_id": call_id,
                                "tool": name,
                                "title": tool_title,
                                "error": policy_decision.reason,
                                "timestamp": now_ms(),
                            }
                        )
                    return [
                        {
                            "type": "text",
                            "text": f"Tool policy denied: {policy_decision.reason}",
                        }
                    ]
                verification_plan = build_verification_plan(name, args, operation_id)
                if not verification_plan.get("supported"):
                    reason = (
                        "Mutation blocked: deterministic authorization and verification "
                        f"adapters are not registered for {name}."
                    )
                    if publish_events:
                        await self.sse_publish(
                            {
                                "type": "tool.error",
                                "run_id": run_id,
                                "call_id": call_id,
                                "tool": name,
                                "title": tool_title,
                                "error": reason,
                                "timestamp": now_ms(),
                            }
                        )
                    return [{"type": "text", "text": reason}]

            if needs_approval:
                decision = None
                try:
                    decision = (context or {}).get("approval_decisions", {}).get(call_id)
                except Exception:
                    decision = None

                if decision is None:
                    if publish_events:
                        await self.sse_publish(
                            {
                                "type": "tool.awaiting_approval",
                                "run_id": run_id,
                                "call_id": call_id,
                                "tool": name,
                                "title": tool_title,
                                "args": event_args,
                                "context": context or {},
                                "timestamp": now_ms(),
                            }
                        )
                    raise ToolExecutor.ApprovalPending(call_id=call_id, tool=name)
                elif decision is False:
                    if publish_events:
                        await self.sse_publish(
                            {
                                "type": "tool.denied",
                                "call_id": call_id,
                                "tool": name,
                                "title": tool_title,
                                "args": event_args,
                                "run_id": run_id,
                                "timestamp": now_ms(),
                            }
                        )
                    return [{"type": "text", "text": "Tool call was denied by the user"}]
                else:
                    if publish_events:
                        await self.sse_publish(
                            {
                                "type": "tool.approved",
                                "call_id": call_id,
                                "tool": name,
                                "title": tool_title,
                                "args": event_args,
                                "run_id": run_id,
                                "timestamp": now_ms(),
                            }
                        )

            if controlled_mutation:
                try:
                    controlled_operation, created = await self.mutation_journal.prepare(
                        operation_id=operation_id,
                        run_id=run_id,
                        call_id=call_id,
                        tool_name=name,
                        args=args,
                        context=context,
                        policy=policy_decision,
                        verification_plan=verification_plan,
                    )
                    operation_id = str(controlled_operation.id)
                except Exception as journal_error:
                    logger.exception("Failed to prepare mutation journal for %s", name)
                    return [
                        {
                            "type": "text",
                            "text": (
                                "Mutation blocked before execution because the durable journal "
                                f"is unavailable: {journal_error}"
                            ),
                        }
                    ]

                if not created:
                    requires_verification = controlled_operation.verification_status in {
                        VerificationStatus.PENDING,
                        VerificationStatus.VERIFYING,
                        VerificationStatus.INCONCLUSIVE,
                    } and controlled_operation.execution_status in {
                        ExecutionStatus.REPORTED_SUCCESS,
                        ExecutionStatus.UNKNOWN,
                    }
                    if controlled_operation.verification_status == VerificationStatus.PASSED:
                        return [
                            {
                                "type": "text",
                                "text": (
                                    "Duplicate execution suppressed. The existing operation is "
                                    f"already verified: {operation_id}."
                                ),
                            },
                            _mutation_meta_block(
                                controlled_operation,
                                requires_verification=False,
                                title=tool_title,
                            ),
                        ]
                    if (
                        controlled_operation.verification_status
                        == VerificationStatus.NEEDS_REVIEW
                    ):
                        return [
                            {
                                "type": "text",
                                "text": (
                                    "Duplicate mutation execution suppressed. The existing "
                                    f"operation requires manual review: {operation_id}."
                                ),
                            },
                            _mutation_meta_block(
                                controlled_operation,
                                requires_verification=False,
                                title=tool_title,
                            ),
                        ]
                    if controlled_operation.execution_status in {
                        ExecutionStatus.REPORTED_SUCCESS,
                        ExecutionStatus.UNKNOWN,
                    }:
                        return [
                            {
                                "type": "text",
                                "text": (
                                    "Duplicate mutation execution suppressed; the existing "
                                    f"operation will be verified: {operation_id}."
                                ),
                            },
                            _mutation_meta_block(
                                controlled_operation,
                                requires_verification=requires_verification,
                                title=tool_title,
                            ),
                        ]
                    if controlled_operation.execution_status != ExecutionStatus.PREPARED:
                        return [
                            {
                                "type": "text",
                                "text": (
                                    "Duplicate mutation execution suppressed because operation "
                                    f"{operation_id} is "
                                    f"{controlled_operation.execution_status.value}."
                                ),
                            }
                        ]

                controlled_operation = await self.mutation_journal.mark_executing(
                    operation_id, lease_owner=run_id
                )
                args = attach_operation_metadata(
                    name,
                    args,
                    operation_id,
                    controlled_operation.desired_state_hmac or "",
                )
                if publish_events:
                    await self.sse_publish(
                        {
                            "type": "mutation.prepared",
                            "run_id": run_id,
                            "call_id": call_id,
                            "operation_id": operation_id,
                            "tool": name,
                            "timestamp": now_ms(),
                        }
                    )

            mcp_client = await self._get_mcp_client()

            if publish_events:
                await self.sse_publish(
                    {
                        "type": "tool.executing",
                        "call_id": call_id,
                        "tool": name,
                        "title": tool_title,
                        "args": event_args,
                        "run_id": run_id,
                        "timestamp": now_ms(),
                    }
                )

            try:
                result = await mcp_client.call_tool(
                    tool_name=name, parameters=args, conversation_id=run_id
                )
            except asyncio.CancelledError:
                if controlled_operation is not None:
                    try:
                        await asyncio.shield(
                            self.mutation_journal.record_execution(
                                operation_id,
                                status=ExecutionStatus.UNKNOWN,
                                error={
                                    "error_type": "cancelled",
                                    "ambiguous_outcome": True,
                                },
                            )
                        )
                    except Exception:
                        # Stop convergence or lease recovery may have moved the row
                        # first.  Never turn that race into a swallowed cancellation.
                        logger.exception(
                            "Failed to record ambiguous cancellation for %s", operation_id
                        )
                raise

            tool_had_error = bool(isinstance(result, dict) and result.get("isError", False))
            if tool_had_error:
                error_message = "Tool execution failed"
                if isinstance(result, dict) and result.get("content"):
                    parts: List[str] = []
                    for block in result["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                    if parts:
                        error_message = "\n".join(parts)

                mutation_meta = None
                if controlled_operation is not None:
                    reliability = _reliability_dict(result)
                    # Once an external mutation has entered EXECUTING, a non-zero
                    # command result cannot prove that no side effect occurred.
                    # This is especially important for multi-document kubectl
                    # apply, where earlier resources may already be committed.
                    reliability = {
                        **reliability,
                        "ambiguous_outcome": True,
                        "external_execution_started": True,
                    }
                    outcome = ExecutionStatus.UNKNOWN
                    controlled_operation = await self.mutation_journal.record_execution(
                        operation_id,
                        status=outcome,
                        error={
                            "error_hmac": hmac_fingerprint(
                                settings.INTERNAL_API_KEY, error_message
                            ),
                            "reliability": reliability,
                        },
                    )
                    mutation_meta = _mutation_meta_block(
                        controlled_operation,
                        requires_verification=outcome == ExecutionStatus.UNKNOWN,
                        title=tool_title,
                        display_result=[
                            {
                                "type": "text",
                                "text": (
                                    "The command result was ambiguous. Skyflo is verifying the "
                                    "real infrastructure state and will not repeat the mutation."
                                ),
                            }
                        ],
                    )
                    if publish_events and outcome == ExecutionStatus.UNKNOWN:
                        await self.sse_publish(
                            {
                                "type": "mutation.outcome",
                                "run_id": run_id,
                                "call_id": call_id,
                                "operation_id": operation_id,
                                "tool": name,
                                "execution_status": outcome.value,
                                "timestamp": now_ms(),
                            }
                        )

                if publish_events and (
                    controlled_operation is None
                    or controlled_operation.execution_status != ExecutionStatus.UNKNOWN
                ):
                    await self.sse_publish(
                        {
                            "type": "tool.error",
                            "call_id": call_id,
                            "tool": name,
                            "title": tool_title,
                            "error": error_message,
                            "reliability": result.get("reliability"),
                            "run_id": run_id,
                            "timestamp": now_ms(),
                        }
                    )
                blocks = [{"type": "text", "text": f"Tool error: {error_message}"}]
                if mutation_meta:
                    blocks.append(mutation_meta)
                return blocks

            content_blocks: List[Dict[str, Any]] = []
            if isinstance(result, dict):
                if "content" in result:
                    content_blocks = result["content"]
                elif "result" in result:
                    actual = result["result"]
                    if isinstance(actual, str):
                        content_blocks.append({"type": "text", "text": actual})
                    elif isinstance(actual, dict):
                        content_blocks.append(
                            {"type": "text", "text": json.dumps(actual, indent=2)}
                        )
                    elif isinstance(actual, list):
                        for item in actual:
                            if isinstance(item, dict) and "type" in item:
                                content_blocks.append(item)
                            else:
                                content_blocks.append({"type": "text", "text": str(item)})
                    else:
                        content_blocks.append({"type": "text", "text": str(actual)})
                else:
                    content_blocks.append({"type": "text", "text": json.dumps(result, indent=2)})
            else:
                content_blocks.append({"type": "text", "text": str(result)})

            if controlled_operation is not None:
                controlled_operation = await self.mutation_journal.record_execution(
                    operation_id,
                    status=ExecutionStatus.REPORTED_SUCCESS,
                    result={
                        "result_hmac": hmac_fingerprint(
                            settings.INTERNAL_API_KEY, content_blocks
                        ),
                        "content_block_count": len(content_blocks),
                    },
                )
                content_blocks.append(
                    _mutation_meta_block(
                        controlled_operation,
                        requires_verification=True,
                        title=tool_title,
                        display_result=list(content_blocks),
                    )
                )

            if publish_events and controlled_operation is None:
                await self.sse_publish(
                    {
                        "type": "tool.result",
                        "call_id": call_id,
                        "tool": name,
                        "title": tool_title,
                        "result": content_blocks,
                        "reliability": (
                            result.get("reliability") if isinstance(result, dict) else None
                        ),
                        "run_id": run_id,
                        "timestamp": now_ms(),
                    }
                )
            elif publish_events:
                await self.sse_publish(
                    {
                        "type": "mutation.outcome",
                        "call_id": call_id,
                        "operation_id": operation_id,
                        "tool": name,
                        "title": tool_title,
                        "execution_status": ExecutionStatus.REPORTED_SUCCESS.value,
                        "run_id": run_id,
                        "timestamp": now_ms(),
                    }
                )

            return content_blocks

        except ToolExecutor.ApprovalPending as awaiting:
            raise awaiting
        except Exception as e:
            logger.exception(f"Error executing tool {name}: {e}")
            tool_meta = locals().get("tool_metadata")
            if self.sse_publish and not self._is_system_tool(tool_meta):
                await self.sse_publish(
                    {
                        "type": "tool.error",
                        "call_id": call_id,
                        "tool": name,
                        "title": locals().get("tool_title", name),
                        "error": str(e),
                        "run_id": run_id,
                        "timestamp": now_ms(),
                    }
                )
            return [{"type": "text", "text": f"Error executing {name}: {e}"}]

    async def list_tools(self, category: Optional[str] = None) -> Dict[str, Any]:
        try:
            all_tools = await self._tools.get_all(self._fetch_tools_from_server)
            if category:
                c = category.lower()
                filtered = [
                    t
                    for t in all_tools
                    if isinstance(t.get("name"), str) and c in t["name"].lower()
                ]
                return {"tools": filtered}
            return {"tools": all_tools}
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return {"tools": [], "error": str(e)}

    async def _validate_tool_parameters(
        self,
        name: str,
        args: Dict[str, Any],
        tool_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            tool_schema = tool_metadata or await self._get_tool_metadata(name)
            if not tool_schema:
                return f"Tool '{name}' not found in available tools"

            input_schema = tool_schema.get("inputSchema") or tool_schema.get("input_schema")
            if input_schema and "required" in input_schema:
                required: List[str] = list(input_schema["required"])
                if required:
                    missing = [p for p in required if p not in args]
                    if missing:
                        return f"Missing required parameters: {', '.join(missing)}"

            return None
        except Exception as e:
            logger.error(f"Error validating tool parameters: {e}")
            return f"Parameter validation error: {e}"
