import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlsplit

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from ..config import settings

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self):
        self.mcp_url = settings.MCP_SERVER_URL.rstrip("/")
        self._client: Optional[Client] = None

    def _get_client(self, run_id: Optional[str] = None) -> Client:
        headers = {"X-Internal-API-Key": settings.INTERNAL_API_KEY}
        if run_id:
            headers["X-Skyflo-Run-ID"] = run_id
        transport = StreamableHttpTransport(url=self.mcp_url, headers=headers)
        return Client(transport)

    def _mcp_origin(self) -> str:
        parsed = urlsplit(self.mcp_url)
        return f"{parsed.scheme}://{parsed.netloc}"

    async def __aenter__(self) -> "MCPClient":
        self._client = self._get_client()
        try:
            await self._client.__aenter__()
        except Exception:
            try:
                await self._client.__aexit__(None, None, None)
            finally:
                self._client = None
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(exc_type, exc_val, exc_tb)
            except Exception as e:
                logger.error(f"Error closing MCP client: {e}")
            finally:
                self._client = None

    async def list_tools_raw(self) -> List[Dict[str, Any]]:
        if self._client is None:
            client = self._get_client()
            async with client:
                tools = await client.list_tools()
                return [t.model_dump() for t in tools]

        tools = await self._client.list_tools()
        return [t.model_dump() for t in tools]

    def _get_tool_name(self, tool: Any) -> str:
        if isinstance(tool, dict):
            return str(tool.get("name", ""))
        return str(getattr(tool, "name", ""))

    async def get_tools(self, category: Optional[str] = None) -> Dict[str, Any]:
        try:
            tools = await self.list_tools_raw()
            if category:
                c = category.lower()
                tools = [t for t in tools if c in self._get_tool_name(t).lower()]
            return {"tools": tools}
        except Exception as e:
            logger.error(f"Error fetching tools: {e}")
            return {"tools": []}

    def _parse_content_item(self, content_item: Any) -> Tuple[Dict[str, Any], bool]:
        cd = content_item.model_dump() if hasattr(content_item, "model_dump") else content_item

        if cd.get("type") != "text":
            return cd, False

        text_content = cd.get("text", "")

        def parsed_tool_output(payload: Dict[str, Any], fallback_text: str = ""):
            block: Dict[str, Any] = {
                "type": "text",
                "text": payload.get("output", fallback_text),
            }
            reliability_keys = (
                "error_type",
                "retryable",
                "attempts",
                "duration_ms",
                "fallback_used",
                "ambiguous_outcome",
                "external_execution_started",
            )
            reliability = {key: payload[key] for key in reliability_keys if key in payload}
            if reliability:
                block["reliability"] = reliability
            return block, bool(payload.get("error"))

        if isinstance(text_content, dict) and "output" in text_content and "error" in text_content:
            return parsed_tool_output(text_content)

        if isinstance(text_content, str):
            try:
                parsed = json.loads(text_content)
                if isinstance(parsed, dict) and "output" in parsed and "error" in parsed:
                    return parsed_tool_output(parsed, text_content)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        return cd, False

    def _parse_tool_result(self, result: Any) -> Dict[str, Any]:
        is_error = result.isError or False
        content_blocks: List[Dict[str, Any]] = []
        reliability: List[Dict[str, Any]] = []

        for content_item in result.content:
            parsed_item, item_is_error = self._parse_content_item(content_item)
            is_error = is_error or item_is_error
            content_blocks.append(parsed_item)
            if isinstance(parsed_item.get("reliability"), dict):
                reliability.append(parsed_item["reliability"])

        parsed_result = {
            "content": content_blocks,
            "isError": is_error,
        }
        if reliability:
            parsed_result["reliability"] = reliability[0] if len(reliability) == 1 else reliability
        return parsed_result

    async def call_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        action: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        try:
            inferred_parameters = parameters.copy()
            if (
                action
                and tool_name == "get_resources"
                and "resource_type" not in inferred_parameters
            ):
                inferred_parameters["resource_type"] = {
                    "get_pods": "pod",
                    "get_deployments": "deployment",
                    "get_services": "service",
                    "get_namespaces": "namespace",
                    "get_nodes": "node",
                }.get(action, inferred_parameters.get("resource_type"))

            if conversation_id:
                client = self._get_client(run_id=conversation_id)
                async with client:
                    result = await client.call_tool_mcp(
                        name=tool_name, arguments=inferred_parameters
                    )
                    return self._parse_tool_result(result)
            if self._client is None:
                client = self._get_client()
                async with client:
                    result = await client.call_tool_mcp(
                        name=tool_name, arguments=inferred_parameters
                    )
                    return self._parse_tool_result(result)
            else:
                result = await self._client.call_tool_mcp(
                    name=tool_name, arguments=inferred_parameters
                )
                return self._parse_tool_result(result)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            return {
                "content": [
                    {
                        "type": "text",
                        "text": "An internal error occurred while calling the tool.",
                    }
                ],
                "isError": True,
                "reliability": {
                    "error_type": "transport_error",
                    "retryable": False,
                    "ambiguous_outcome": True,
                },
            }

    async def cancel_run(self, run_id: str) -> Dict[str, Any]:
        url = f"{self._mcp_origin()}/internal/runs/{quote(run_id, safe='')}/cancel"
        headers = {"X-Internal-API-Key": settings.INTERNAL_API_KEY}
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, headers=headers)
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, dict) else {"matched": 0}
        except Exception as e:
            logger.error("Failed to cancel MCP processes for run %s: %s", run_id, e)
            return {"matched": 0, "error": "mcp_cancel_unavailable"}
