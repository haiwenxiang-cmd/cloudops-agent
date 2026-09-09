from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from src.api.services.mcp_client import MCPClient


class MCPClientReliabilityTests(IsolatedAsyncioTestCase):
    def test_run_id_and_internal_key_are_transport_headers(self) -> None:
        transport = MagicMock()
        with patch(
            "src.api.services.mcp_client.StreamableHttpTransport",
            return_value=transport,
        ) as transport_factory:
            with patch("src.api.services.mcp_client.Client"):
                MCPClient()._get_client(run_id="run-123")

        _, kwargs = transport_factory.call_args
        self.assertEqual(kwargs["headers"]["X-Skyflo-Run-ID"], "run-123")
        self.assertTrue(kwargs["headers"]["X-Internal-API-Key"])

    async def test_cancel_run_calls_authenticated_internal_route(self) -> None:
        response = MagicMock()
        response.json.return_value = {"matched": 1, "terminated": 1, "killed": 0}
        http_client = AsyncMock()
        http_client.post.return_value = response
        context = AsyncMock()
        context.__aenter__.return_value = http_client

        with patch("src.api.services.mcp_client.httpx.AsyncClient", return_value=context):
            result = await MCPClient().cancel_run("b57c9cb2-4a42-4eb7-824f-91dbcbacb27d")

        self.assertEqual(result["terminated"], 1)
        url = http_client.post.await_args.args[0]
        headers = http_client.post.await_args.kwargs["headers"]
        self.assertTrue(url.endswith("/internal/runs/b57c9cb2-4a42-4eb7-824f-91dbcbacb27d/cancel"))
        self.assertTrue(headers["X-Internal-API-Key"])
        response.raise_for_status.assert_called_once_with()

    async def test_cancel_run_degrades_without_raising(self) -> None:
        context = AsyncMock()
        context.__aenter__.side_effect = OSError("mcp unavailable")

        with patch("src.api.services.mcp_client.httpx.AsyncClient", return_value=context):
            result = await MCPClient().cancel_run("b57c9cb2-4a42-4eb7-824f-91dbcbacb27d")

        self.assertEqual(result, {"matched": 0, "error": "mcp_cancel_unavailable"})
