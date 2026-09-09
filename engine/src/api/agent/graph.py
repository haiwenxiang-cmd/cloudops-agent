import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from ..config import settings
from ..memory.events import emit_memory_context_loaded
from ..memory.formatter import MemoryContextFormatter
from ..memory.retrieval import MemoryRetrievalService
from ..memory.schemas import MemoryHit
from ..memory.tools import MEMORY_TOOL_NAMES
from ..services.approvals import ApprovalService
from ..services.checkpointer import get_checkpointer
from ..services.mcp_client import MCPClient
from ..services.mutation_utils import redact_tool_args
from ..services.mutation_verifier import MutationVerifier
from ..services.stop_service import clear_stop
from ..services.tool_executor import AVAILABLE_TOOLSETS, ToolExecutor
from ..utils.clock import now_ms
from ..utils.helpers import get_state_value
from .model_node import ModelNode
from .state import AgentState
from .stop import StopRequested, check_stop

logger = logging.getLogger(__name__)

INTERNAL_META_TOOLS = frozenset({"load_toolset"} | MEMORY_TOOL_NAMES)

_MEMORY_CONTEXT_KEYS = ("_user_id", "_conversation_id", "_run_id")


def _inject_memory_context(state: Dict[str, Any], tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Merge engine-owned context into memory tool args (overrides any LLM-supplied values)."""
    merged = dict(tool_args)
    for key in list(merged):
        if key in _MEMORY_CONTEXT_KEYS or key.lstrip("_") in {
            "user_id",
            "conversation_id",
            "run_id",
        }:
            merged.pop(key, None)

    user_id = get_state_value(state, "user_id")
    conversation_id = get_state_value(state, "conversation_id")
    run_id = get_state_value(state, "run_id")

    if user_id:
        merged["_user_id"] = str(user_id)
    if conversation_id:
        merged["_conversation_id"] = str(conversation_id)
    if run_id:
        merged["_run_id"] = str(run_id)

    return merged


EventCallback = Callable[[Dict[str, Any]], Awaitable[None]]


def route_after_model(state: Dict[str, Any]) -> Literal["gate", "verification", "final"]:
    pending_tools = get_state_value(state, "pending_tools", [])

    if pending_tools:
        return "gate"

    if get_state_value(state, "verification_required", False):
        return "verification"

    return "final"


def route_from_entry(
    state: Dict[str, Any],
) -> Literal["gate", "verification", "memory_prepare"]:
    pending_tools = get_state_value(state, "pending_tools", [])
    if pending_tools:
        return "gate"
    if get_state_value(state, "verification_required", False):
        return "verification"
    return "memory_prepare"


def route_after_gate(state: Dict[str, Any]) -> Literal["model", "verification", "final"]:
    if get_state_value(state, "awaiting_approval", False):
        return "final"
    if get_state_value(state, "error"):
        return "final"
    if get_state_value(state, "verification_required", False):
        return "verification"
    return "model"


def route_after_verification(state: Dict[str, Any]) -> Literal["model", "final"]:
    if get_state_value(state, "verification_blocked", False):
        return "final"
    if get_state_value(state, "verification_required", False):
        return "final"
    return "model"


class WorkflowGraph:
    def __init__(
        self,
        event_callback: Optional[EventCallback] = None,
    ):
        self.event_callback = event_callback

        self.approval_service = ApprovalService()
        self.mcp_client = MCPClient()
        self.tool_executor = ToolExecutor(
            approvals=self.approval_service,
            sse_publish=self.event_callback,
            mcp_client=self.mcp_client,
            owns_client=False,
        )
        self.mutation_verifier = MutationVerifier(
            mcp_client=self.mcp_client,
            journal=self.tool_executor.mutation_journal,
        )
        self.model_node = ModelNode(
            event_callback=self.event_callback,
            tools_provider=self.tool_executor.get_llm_compatible_tools,
        )
        self._memory_retrieval = MemoryRetrievalService()
        self._memory_formatter = MemoryContextFormatter()
        self.graph = self._build_graph()
        self.compiled_graph = None
        self.checkpointer = None

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("entry", self._entry_node)
        workflow.add_node("memory_prepare", self._memory_prepare_node)
        workflow.add_node("model", self._model_node)
        workflow.add_node("gate", self._gate_node)
        workflow.add_node("verification", self._verification_node)
        workflow.add_node("final", self._final_node)

        workflow.add_edge(START, "entry")
        workflow.add_conditional_edges(
            "entry",
            route_from_entry,
            {
                "gate": "gate",
                "verification": "verification",
                "memory_prepare": "memory_prepare",
            },
        )
        workflow.add_edge("memory_prepare", "model")
        workflow.add_conditional_edges(
            "model",
            route_after_model,
            {"gate": "gate", "verification": "verification", "final": "final"},
        )
        workflow.add_conditional_edges(
            "gate",
            route_after_gate,
            {"model": "model", "verification": "verification", "final": "final"},
        )
        workflow.add_conditional_edges(
            "verification", route_after_verification, {"model": "model", "final": "final"}
        )
        workflow.add_edge("final", END)

        return workflow

    async def _compile_graph(self):
        checkpointer = None

        if settings.ENABLE_POSTGRES_CHECKPOINTER:
            try:
                checkpointer = get_checkpointer()
            except Exception as e:
                logger.warning(
                    f"Failed to get shared checkpointer: {e}. "
                    f"Falling back to in-memory checkpointer"
                )
                checkpointer = None

        if checkpointer is None:
            checkpointer = MemorySaver()
            logger.debug("Graph compiled with in-memory checkpointer")

        self.checkpointer = checkpointer

        compiled = self.graph.compile(checkpointer=checkpointer)
        return compiled

    async def _ensure_compiled(self):
        if self.compiled_graph is None:
            self.compiled_graph = await self._compile_graph()

    async def _entry_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        await check_stop(state)
        updates: Dict[str, Any] = {
            "ttft_emitted": False,
            "memory_context_loaded": False,
            "memory_context_msg": None,
            "error": None,
            "verification_blocked": False,
        }
        if get_state_value(state, "awaiting_approval", False):
            updates["awaiting_approval"] = False
        return updates

    async def _memory_prepare_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not settings.MEMORY_ENABLED:
            return {}

        # Skip if context already loaded for this turn
        if get_state_value(state, "memory_context_loaded", False):
            return {}

        # Find the latest user message to use as the retrieval query
        messages = get_state_value(state, "messages", [])
        latest_user_msg = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            latest_user_msg = part.get("text", "")
                            break
                elif isinstance(content, str):
                    latest_user_msg = content
                if latest_user_msg:
                    break

        if not latest_user_msg:
            return {"memory_context_loaded": True}

        user_id_str = get_state_value(state, "user_id")
        conversation_id_str = get_state_value(state, "conversation_id")
        run_id = get_state_value(state, "run_id", "unknown")

        try:
            from uuid import UUID

            if isinstance(user_id_str, UUID):
                user_id = user_id_str
            elif user_id_str:
                user_id = UUID(str(user_id_str))
            else:
                user_id = None
        except (ValueError, AttributeError, TypeError):
            user_id = None

        try:
            hits: List[MemoryHit] = await self._memory_retrieval.retrieve_for_turn(
                query=latest_user_msg,
                user_id=user_id,
                conversation_id=conversation_id_str,
                run_id=run_id,
                max_docs=settings.MEMORY_CONTEXT_MAX_DOCS,
                token_budget=settings.MEMORY_CONTEXT_TOKEN_BUDGET,
            )
        except Exception as e:
            logger.warning(f"Memory retrieval failed (non-fatal): {e}")
            return {"memory_context_loaded": True}

        if not hits:
            return {"memory_context_loaded": True}

        context_msg = self._memory_formatter.format(hits)

        try:
            await emit_memory_context_loaded(self.event_callback, run_id, hits)
        except Exception as e:
            logger.debug(f"Failed to emit memory.context.loaded: {e}")

        updates: Dict[str, Any] = {
            "memory_context_loaded": True,
            "memory_hits": [h.minimal() for h in hits],
        }

        if context_msg:
            # Store in a dedicated field so the model node can inject it before
            # the last user message rather than appending at the end via operator.add.
            updates["memory_context_msg"] = context_msg

        return updates

    async def _model_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await check_stop(state)
            result = await self.model_node(state)

            updated_state = {}
            if "messages" in result:
                updated_state["messages"] = result["messages"]
            if "pending_tools" in result:
                updated_state["pending_tools"] = result["pending_tools"]
            if "error" in result:
                updated_state["error"] = result["error"]
            if "ttft_emitted" in result:
                updated_state["ttft_emitted"] = result["ttft_emitted"]

            return updated_state

        except StopRequested:
            raise
        except Exception as e:
            logger.exception(f"Error in model node: {str(e)}")
            return {"error": str(e)}

    async def _gate_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        try:
            await check_stop(state)

            pending_tools = get_state_value(state, "pending_tools", [])
            if not pending_tools:
                return {"pending_tools": [], "suppress_pending_event": False}

            try:
                suppress_pending_event = get_state_value(state, "suppress_pending_event", False)
                if self.event_callback and not suppress_pending_event:
                    pending_payload = []
                    for tool_call in pending_tools:
                        try:
                            tool_name = tool_call.get("name", "unknown")
                            display_id = tool_call.get("call_id") or tool_call.get("id")
                            tool_args = (
                                tool_call.get("args", {}) if isinstance(tool_call, dict) else {}
                            )

                            if tool_name in INTERNAL_META_TOOLS:
                                continue

                            title_value = tool_name
                            try:
                                metadata = await self.tool_executor._get_tool_metadata(tool_name)
                                if metadata and isinstance(metadata, dict):
                                    title_value = metadata.get("title", tool_name)
                            except Exception:
                                title_value = tool_name

                            requires_approval_value = False
                            try:
                                requires_approval_value = await self.approval_service.need_approval(
                                    tool_name, tool_args
                                )
                            except Exception:
                                requires_approval_value = True

                            pending_payload.append(
                                {
                                    "call_id": display_id,
                                    "tool": tool_name,
                                    "title": title_value,
                                    "args": redact_tool_args(tool_name, tool_args),
                                    "requires_approval": requires_approval_value,
                                    "timestamp": now_ms(),
                                }
                            )
                        except Exception:
                            continue

                    if pending_payload:
                        await self.event_callback(
                            {
                                "type": "tools.pending",
                                "run_id": get_state_value(state, "run_id", "unknown"),
                                "tools": pending_payload,
                                "timestamp": now_ms(),
                            }
                        )
            except Exception:
                pass

            tool_messages = []
            mutation_operations = list(get_state_value(state, "mutation_operations", []))
            # A Gate invocation may be resuming a checkpoint after a later tool in
            # the same batch paused for approval.  Rebuild this invariant from the
            # durable per-operation metadata instead of resetting it for the new
            # invocation; otherwise an earlier executed mutation can skip
            # verification when the resumed tool is denied or is not controlled.
            verification_required = any(
                bool(item.get("requires_verification"))
                for item in mutation_operations
                if isinstance(item, dict)
            )
            loaded_toolsets = dict(get_state_value(state, "loaded_toolsets", {"k8s": False}))
            toolsets_changed = False

            for idx, tool_call in enumerate(pending_tools):
                try:
                    tool_name = tool_call["name"]
                    original_id = tool_call.get("id")
                    display_id = tool_call.get("call_id") or original_id
                    tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else {}

                    if tool_name == "load_toolset":
                        result_content = self._handle_load_toolset(tool_args, loaded_toolsets)
                        toolsets_changed = True
                        tool_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": original_id,
                                "name": tool_name,
                                "content": result_content,
                            }
                        )
                        continue

                    if tool_name in MEMORY_TOOL_NAMES:
                        tool_args = _inject_memory_context(state, tool_args)

                    tool_results = await self.tool_executor.execute(
                        run_id=get_state_value(state, "run_id", "unknown"),
                        name=tool_name,
                        args=tool_args,
                        context={
                            "conversation_id": get_state_value(state, "conversation_id"),
                            "user_id": get_state_value(state, "user_id"),
                            "user_role": get_state_value(state, "user_role"),
                            "environment": get_state_value(
                                state, "environment", settings.SKYFLO_ENVIRONMENT
                            ),
                            "approval_decisions": get_state_value(state, "approval_decisions", {}),
                            "approval_reasons": get_state_value(state, "approval_reasons", {}),
                        },
                        call_id=display_id,
                        operation_id=tool_call.get("operation_id"),
                    )

                    result_content = ""
                    for block in tool_results:
                        if block.get("type") == "skyflo.mutation":
                            mutation_operations = [
                                item
                                for item in mutation_operations
                                if item.get("operation_id") != block.get("operation_id")
                            ]
                            mutation_operations.append(block)
                            verification_required = verification_required or bool(
                                block.get("requires_verification")
                            )
                            continue
                        if block.get("type") == "text":
                            result_content += block.get("text", "")
                        else:
                            result_content += str(block)

                    tool_message = {
                        "role": "tool",
                        "tool_call_id": original_id,
                        "name": tool_name,
                        "content": result_content,
                    }
                    tool_messages.append(tool_message)

                except ToolExecutor.ApprovalPending:
                    remaining_tools = pending_tools[idx:]
                    result = {
                        "messages": tool_messages,
                        "pending_tools": remaining_tools,
                        "awaiting_approval": True,
                        "suppress_pending_event": False,
                        "mutation_operations": mutation_operations,
                        "verification_required": verification_required,
                        "verification_blocked": False,
                    }
                    if toolsets_changed:
                        result["loaded_toolsets"] = loaded_toolsets
                    return result
                except Exception as tool_error:
                    err_tool = tool_call.get("name", "unknown")
                    logger.exception(f"Error executing tool {err_tool}: {tool_error}")

                    error_message = {
                        "role": "tool",
                        "tool_call_id": original_id,
                        "name": err_tool,
                        "content": f"Error executing tool {err_tool}: {tool_error}",
                    }
                    tool_messages.append(error_message)

            result = {
                "messages": tool_messages,
                "pending_tools": [],
                "awaiting_approval": False,
                "suppress_pending_event": False,
                "mutation_operations": mutation_operations,
                "verification_required": verification_required,
                "verification_blocked": False,
            }
            if toolsets_changed:
                result["loaded_toolsets"] = loaded_toolsets
            return result

        except Exception as e:
            logger.exception(f"Error in gate node: {str(e)}")

            error_message = {
                "role": "assistant",
                "name": None,
                "content": f"Error in tool execution gate: {str(e)}",
            }

            return {
                "messages": [error_message],
                "pending_tools": [],
                "error": str(e),
                "suppress_pending_event": False,
            }

    async def _verification_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Enforce postconditions without asking the model to remember to verify."""
        await check_stop(state)
        pending = [
            item
            for item in get_state_value(state, "mutation_operations", [])
            if item.get("requires_verification")
        ]
        if not pending:
            return {"verification_required": False, "verification_blocked": False}

        evidence: List[Dict[str, Any]] = []
        updated_operations = list(get_state_value(state, "mutation_operations", []))
        for item in pending:
            operation_id = item.get("operation_id")
            if self.event_callback:
                await self.event_callback(
                    {
                        "type": "mutation.verifying",
                        "run_id": get_state_value(state, "run_id"),
                        "call_id": item.get("call_id"),
                        "operation_id": operation_id,
                        "tool": item.get("tool"),
                        "timestamp": now_ms(),
                    }
                )
            try:
                result = await self.mutation_verifier.verify(
                    operation_id,
                    lease_owner=f"foreground:{get_state_value(state, 'run_id', 'unknown')}",
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Deterministic verification failed for %s", operation_id)
                result = {
                    "passed": False,
                    "inconclusive": True,
                    "status": "inconclusive",
                    "operation": {"operation_id": operation_id},
                    "evidence": {
                        "reason": "verification_runtime_error",
                        "detail": type(exc).__name__,
                    },
                }
            evidence.append(result)
            replacement = dict(item)
            verification_status = result.get("status", "inconclusive")
            replacement["verification_status"] = verification_status
            replacement["requires_verification"] = verification_status in {
                "pending",
                "verifying",
                "inconclusive",
            }
            updated_operations = [
                replacement if op.get("operation_id") == operation_id else op
                for op in updated_operations
            ]
            if self.event_callback:
                await self.event_callback(
                    {
                        "type": (
                            "mutation.verified"
                            if result.get("passed")
                            else (
                                "mutation.manual_review_required"
                                if result.get("status") == "needs_review"
                                else
                                "mutation.verification_failed"
                                if result.get("status") == "failed"
                                else "mutation.inconclusive"
                            )
                        ),
                        "run_id": get_state_value(state, "run_id"),
                        "operation_id": operation_id,
                        "tool": item.get("tool"),
                        "verification_status": result.get("status"),
                        "evidence": result.get("evidence"),
                        "timestamp": now_ms(),
                    }
                )
                if result.get("passed"):
                    await self.event_callback(
                        {
                            "type": "tool.result",
                            "run_id": get_state_value(state, "run_id"),
                            "call_id": item.get("call_id"),
                            "tool": item.get("tool"),
                            "title": item.get("title") or item.get("tool"),
                            "result": item.get("display_result")
                            or [
                                {
                                    "type": "text",
                                    "text": (
                                        "Mutation completed and deterministic runtime "
                                        "verification passed."
                                    ),
                                }
                            ],
                            "timestamp": now_ms(),
                        }
                    )
                else:
                    await self.event_callback(
                        {
                            "type": "tool.error",
                            "run_id": get_state_value(state, "run_id"),
                            "call_id": item.get("call_id"),
                            "tool": item.get("tool"),
                            "title": item.get("title") or item.get("tool"),
                            "error": (
                                "Mutation outcome could not be verified; success reporting "
                                "is blocked."
                            ),
                            "timestamp": now_ms(),
                        }
                    )

        all_passed = all(item.get("passed") for item in evidence)
        if all_passed:
            summary = {
                "role": "system",
                "content": (
                    "Skyflo deterministic runtime verification passed for mutation operations: "
                    + ", ".join(item["operation"]["operation_id"] for item in evidence)
                    + ". You may report success using this evidence."
                ),
            }
            return {
                "messages": [summary],
                "mutation_operations": updated_operations,
                "verification_required": False,
                "verification_blocked": False,
                "verification_evidence": evidence,
                "error": None,
            }

        failed_ids = [
            item.get("operation", {}).get("operation_id", "unknown")
            for item in evidence
            if not item.get("passed")
        ]
        message = (
            "Mutation result was not verified. Skyflo runtime blocks any success claim. "
            "Operation IDs: "
            + ", ".join(failed_ids)
        )
        if self.event_callback:
            await self.event_callback(
                {
                    "type": "token",
                    "text": message,
                    "conversation_id": get_state_value(state, "conversation_id"),
                    "run_id": get_state_value(state, "run_id"),
                }
            )
            await self.event_callback(
                {
                    "type": "workflow.error",
                    "run_id": get_state_value(state, "run_id"),
                    "error": message,
                    "timestamp": now_ms(),
                }
            )
        still_pending = any(
            bool(item.get("requires_verification")) for item in updated_operations
        )
        return {
            "messages": [{"role": "assistant", "content": message}],
            "mutation_operations": updated_operations,
            "verification_required": still_pending,
            "verification_blocked": True,
            "verification_evidence": evidence,
            "error": message,
        }

    def _handle_load_toolset(
        self,
        args: Dict[str, Any],
        loaded_toolsets: Dict[str, bool],
    ) -> str:
        toolset = (args.get("toolset") or "").strip().lower()

        if toolset not in AVAILABLE_TOOLSETS:
            return f"Unknown toolset '{toolset}'. Available: {', '.join(AVAILABLE_TOOLSETS)}"

        val = args.get("include_write_tools")
        if isinstance(val, bool):
            include_write = val
        elif isinstance(val, str) and val.strip().lower() in ("true", "false"):
            include_write = val.strip().lower() == "true"
        elif val is None:
            return (
                "Missing required argument 'include_write_tools' (boolean). "
                "Pass true to load write/mutation tools, false for read-only."
            )
        else:
            return f"Invalid 'include_write_tools' value: {val!r}. Must be a boolean."

        already_loaded = toolset in loaded_toolsets
        had_write = loaded_toolsets.get(toolset, False)
        upgraded = include_write and not had_write

        loaded_toolsets[toolset] = had_write or include_write

        mode = "read and write" if loaded_toolsets[toolset] else "read-only"

        if already_loaded and not upgraded:
            return f"Toolset '{toolset}' is already loaded ({mode})."

        return f"Toolset '{toolset}' loaded ({mode}). You can now use {toolset} tools."

    async def _final_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        end_time = time.time()
        start_time = get_state_value(state, "start_time", end_time)
        duration = end_time - start_time
        return {"done": True, "end_time": end_time, "duration": duration}

    async def invoke(self, initial_state: Dict[str, Any], **kwargs):
        try:
            await self._ensure_compiled()

            start_time_value = get_state_value(initial_state, "start_time")
            if start_time_value is None:
                current_time = time.time()

                if hasattr(initial_state, "start_time"):
                    initial_state.start_time = current_time
                elif hasattr(initial_state, "__setitem__"):
                    initial_state["start_time"] = current_time

            config = kwargs.get("config", {})
            if "configurable" not in config:
                thread_id = get_state_value(initial_state, "conversation_id") or get_state_value(
                    initial_state, "run_id", "default"
                )
                config["configurable"] = {"thread_id": thread_id}
                config["recursion_limit"] = settings.LLM_MAX_ITERATIONS
                kwargs["config"] = config

            try:
                result = await self.compiled_graph.ainvoke(initial_state, **kwargs)
                return result
            except StopRequested:
                end_time = time.time()
                start_time = get_state_value(initial_state, "start_time", end_time)
                duration = end_time - start_time
                try:
                    await clear_stop(get_state_value(initial_state, "run_id"))
                except Exception:
                    pass
                return {"done": True, "stopped": True, "duration": duration}
            except GraphRecursionError:
                error_message = (
                    f"The AI Agent has reached the maximum number of iterations "
                    f"of {settings.LLM_MAX_ITERATIONS} for the current prompt. "
                    f"You can continue the conversation. If you want to update "
                    f"the max iterations, update the LLM_MAX_ITERATIONS "
                    f"environment variable."
                )
                if self.event_callback:
                    await self.event_callback(
                        {
                            "type": "workflow.error",
                            "run_id": get_state_value(initial_state, "run_id"),
                            "error": error_message,
                        }
                    )
                return {"done": True, "error": error_message}
            except Exception as e:
                error_message = f"An unknown error occurred while executing the workflow: {e}"
                if self.event_callback:
                    await self.event_callback(
                        {
                            "type": "workflow.error",
                            "run_id": get_state_value(initial_state, "run_id"),
                            "error": error_message,
                        }
                    )
                return {"done": True, "error": error_message}

        except Exception as e:
            error_message = f"An unknown error occurred while executing the workflow: {e}"
            if self.event_callback:
                await self.event_callback(
                    {
                        "type": "workflow.error",
                        "run_id": get_state_value(initial_state, "run_id"),
                        "error": error_message,
                    }
                )
            return {"done": True, "error": error_message}

    async def close(self):
        try:
            await self.tool_executor.close()
            await self.approval_service.close()

        except Exception as e:
            logger.error(f"Error closing workflow resources: {str(e)}")


def build_graph(
    event_callback: Optional[EventCallback] = None,
) -> WorkflowGraph:
    return WorkflowGraph(event_callback=event_callback)
