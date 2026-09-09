import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from .config import close_db_connection, init_db, settings
from .endpoints import api_router
from .endpoints.agent import publish_event, remember_run_conversation
from .memory.seed import seed_memory_stores
from .middleware import setup_middleware
from .services.checkpointer import close_graph_checkpointer, init_graph_checkpointer
from .services.conversation_persistence import ConversationPersistenceService
from .services.limiter import close_limiter, init_limiter
from .services.mcp_client import MCPClient
from .services.mutation_verifier import MutationRecoveryWorker
from .utils.clock import now_ms

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

MCP_RETRY_ATTEMPTS = 5
MCP_RETRY_DELAY = 3


async def publish_recovered_mutation(operation, result) -> None:
    """Bridge durable background reconciliation back to chat state and events."""
    passed = bool(result.get("passed"))
    status = str(result.get("status") or "inconclusive")
    mutation_event_type = (
        "mutation.verified"
        if passed
        else "mutation.manual_review_required"
        if status == "needs_review"
        else "mutation.verification_failed"
        if status == "failed"
        else "mutation.inconclusive"
    )
    text = (
        "Mutation completed and background verification passed."
        if passed
        else "Mutation verification exhausted its bounded retry budget and requires manual review."
        if status == "needs_review"
        else f"Background mutation verification finished with status: {status}."
    )
    result_blocks = [{"type": "text", "text": text}]

    if operation.conversation_id and operation.call_id:
        try:
            await ConversationPersistenceService().update_tool_segment_status(
                conversation_id=str(operation.conversation_id),
                call_id=str(operation.call_id),
                status="completed" if passed else "error",
                error=None if passed else text,
                result=result_blocks,
                allow_cancelled_transition=True,
            )
        except Exception:
            logger.exception(
                "Failed to persist recovered mutation %s into conversation",
                operation.id,
            )

    if operation.run_id:
        if operation.conversation_id:
            await remember_run_conversation(
                str(operation.run_id), str(operation.conversation_id)
            )
        channel = f"run:{operation.run_id}"
        await publish_event(
            channel,
            mutation_event_type,
            {
                "type": mutation_event_type,
                "run_id": operation.run_id,
                "operation_id": str(operation.id),
                "call_id": operation.call_id,
                "tool": operation.tool_name,
                "verification_status": status,
                "evidence": result.get("evidence") or {},
                "timestamp": now_ms(),
            },
        )
        await publish_event(
            channel,
            "tool.result" if passed else "tool.error",
            {
                "type": "tool.result" if passed else "tool.error",
                "run_id": operation.run_id,
                "call_id": operation.call_id,
                "tool": operation.tool_name,
                **({"result": result_blocks} if passed else {"error": text}),
                "timestamp": now_ms(),
            },
        )
        await publish_event(
            channel,
            "workflow_complete",
            {
                "type": "workflow_complete",
                "run_id": operation.run_id,
                "result": {
                    "done": True,
                    "operation_id": str(operation.id),
                    "verification_status": status,
                },
                "status": "recovered",
                "timestamp": now_ms(),
            },
        )


async def verify_mcp_connection() -> None:
    last_error: Exception | None = None

    for attempt in range(1, MCP_RETRY_ATTEMPTS + 1):
        try:
            client = MCPClient()
            tools = await client.list_tools_raw()
            logger.info(
                "MCP server %s connected successfully. Available tools: %d",
                client.mcp_url,
                len(tools),
            )
            return
        except Exception as e:
            last_error = e
            logger.warning(
                "MCP connection attempt %d/%d failed: %s",
                attempt,
                MCP_RETRY_ATTEMPTS,
                str(e),
            )
            if attempt < MCP_RETRY_ATTEMPTS:
                await asyncio.sleep(MCP_RETRY_DELAY)
    raise RuntimeError(
        f"MCP server unreachable after {MCP_RETRY_ATTEMPTS} attempts. Last error: {last_error}"
    ) from last_error


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.APP_NAME} version {settings.APP_VERSION}")

    if settings.MEMORY_ENABLED and not settings.INTERNAL_API_KEY:
        raise RuntimeError(
            "INTERNAL_API_KEY must be set when MEMORY_ENABLED=true. "
            "Generate with: openssl rand -base64 32. "
            "Set MEMORY_ENABLED=false to run without the memory system."
        )

    await verify_mcp_connection()
    await init_db()
    mutation_recovery = None
    if settings.MUTATION_CONTROL_ENABLED:
        mutation_recovery = MutationRecoveryWorker(on_reconciled=publish_recovered_mutation)
        await mutation_recovery.start()
    await init_limiter()
    await init_graph_checkpointer()
    if settings.MEMORY_ENABLED:
        try:
            await seed_memory_stores()
        except Exception as e:
            logger.warning("Memory store seeding failed (non-fatal): %s", e)

    yield

    logger.info(f"Shutting down {settings.APP_NAME}")
    if mutation_recovery:
        await mutation_recovery.stop()
    await close_db_connection()
    await close_limiter()
    await close_graph_checkpointer()


def create_application() -> FastAPI:
    application = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    setup_middleware(application)

    application.include_router(api_router, prefix=settings.API_V1_STR)

    return application


app = create_application()
