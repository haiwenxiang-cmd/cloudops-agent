import secrets
import shutil
import uuid

from fastmcp import FastMCP
from starlette.responses import JSONResponse

from config.settings import settings
from utils.commands import cancel_run_processes

mcp = FastMCP(
    "Skyflo MCP Server",
    instructions="""
    # Skyflo MCP

    This MCP allows you to:
    1. Manage Kubernetes clusters, resources, and deployments using kubectl operations
    2. Install and manage applications with Helm charts and repositories
    3. Execute progressive deployments with Argo Rollouts (blue/green, canary strategies)
    4. Troubleshoot and diagnose cluster issues with comprehensive validation
    """,
)


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "ok"})


@mcp.custom_route("/health/ready", methods=["GET"])
async def health_ready(request):
    required_tools = ["kubectl", "helm", "kubectl-argo-rollouts"]
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing_tools:
        return JSONResponse({"status": "error", "missing_tools": missing_tools}, status_code=503)
    return JSONResponse({"status": "ready"})


@mcp.custom_route("/internal/runs/{run_id}/cancel", methods=["POST"])
async def cancel_run(request):
    supplied_key = request.headers.get("x-internal-api-key", "")
    if not secrets.compare_digest(supplied_key, settings.INTERNAL_API_KEY):
        return JSONResponse({"detail": "Unauthorized"}, status_code=401)

    raw_run_id = request.path_params.get("run_id", "")
    try:
        run_id = str(uuid.UUID(raw_run_id))
    except (ValueError, TypeError, AttributeError):
        return JSONResponse({"detail": "Invalid run_id"}, status_code=400)

    result = await cancel_run_processes(run_id)
    return JSONResponse({"run_id": run_id, **result})


# Import tool modules to register them with the MCP server
# The @mcp.tool() decorators execute at import time
import tools.argo  # noqa: E402, F401
import tools.helm  # noqa: E402, F401
import tools.jenkins  # noqa: E402, F401
import tools.kubectl  # noqa: E402, F401
import tools.memory  # noqa: E402, F401
