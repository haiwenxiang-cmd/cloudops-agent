"""Reliable, run-aware subprocess execution shared by MCP tools."""

import asyncio
import logging
import os
import signal
import time
from collections import defaultdict
from typing import Optional

from fastmcp.server.dependencies import get_http_request

from config.settings import settings

from .models import ToolOutput

logger = logging.getLogger(__name__)

RUN_ID_HEADER = "x-skyflo-run-id"

_active_processes: dict[str, set[asyncio.subprocess.Process]] = defaultdict(set)
_active_processes_lock = asyncio.Lock()
_cancelled_runs: dict[str, float] = {}

_KUBECTL_READ_ONLY = frozenset(
    {
        "api-resources",
        "api-versions",
        "auth",
        "cluster-info",
        "describe",
        "diff",
        "explain",
        "get",
        "logs",
        "top",
        "version",
    }
)
_KUBECTL_ROLLOUT_READ_ONLY = frozenset({"history", "status"})
_ARGO_READ_ONLY = frozenset({"get", "history", "list", "status"})
_HELM_READ_ONLY = frozenset({"env", "get", "history", "list", "search", "show", "status"})

_TRANSIENT_MARKERS = (
    "connection refused",
    "connection reset",
    "context deadline exceeded",
    "gateway timeout",
    "i/o timeout",
    "no route to host",
    "service unavailable",
    "server is currently unable",
    "temporarily unavailable",
    "tls handshake timeout",
    "transport is closing",
    "unexpected eof",
)
_RATE_LIMIT_MARKERS = ("429", "rate limit", "too many requests", "throttl")
_AUTH_MARKERS = (
    "access denied",
    "authentication required",
    "forbidden",
    "permission denied",
    "unauthorized",
)
_CONFLICT_MARKERS = ("already exists", "conflict", "object has been modified")
_NOT_FOUND_MARKERS = ("not found", "notfound", "no such file or directory")
_INVALID_MARKERS = ("invalid argument", "required flag", "unknown flag", "unknown command")


def _current_run_id() -> Optional[str]:
    try:
        request = get_http_request()
    except RuntimeError:
        return None
    run_id = request.headers.get(RUN_ID_HEADER)
    return run_id.strip() if run_id and run_id.strip() else None


def _is_retry_safe(cmd: str, args: list[str]) -> bool:
    if not args:
        return False

    if cmd == "kubectl":
        if args[:2] == ["argo", "rollouts"]:
            return len(args) > 2 and args[2] in _ARGO_READ_ONLY
        if args[0] == "rollout":
            return len(args) > 1 and args[1] in _KUBECTL_ROLLOUT_READ_ONLY
        return args[0] in _KUBECTL_READ_ONLY

    if cmd == "helm":
        return args[0] in _HELM_READ_ONLY

    return False


def _classify_error(message: str, *, timed_out: bool = False) -> tuple[str, bool]:
    if timed_out:
        return "timeout", True

    normalized = message.lower()
    if any(marker in normalized for marker in _RATE_LIMIT_MARKERS):
        return "rate_limited", True
    if any(marker in normalized for marker in _AUTH_MARKERS):
        return "authorization", False
    if any(marker in normalized for marker in _CONFLICT_MARKERS):
        return "conflict", False
    if any(marker in normalized for marker in _NOT_FOUND_MARKERS):
        return "not_found", False
    if any(marker in normalized for marker in _INVALID_MARKERS):
        return "invalid_input", False
    if any(marker in normalized for marker in _TRANSIENT_MARKERS):
        return "transient", True
    return "command_failed", False


def _prune_cancelled_runs(now: float) -> None:
    ttl = settings.RUN_CANCEL_TOMBSTONE_TTL_SECONDS
    expired = [
        run_id
        for run_id, cancelled_at in _cancelled_runs.items()
        if now - cancelled_at >= ttl
    ]
    for run_id in expired:
        _cancelled_runs.pop(run_id, None)


async def _register_process(run_id: Optional[str], proc: asyncio.subprocess.Process) -> bool:
    if not run_id:
        return True
    async with _active_processes_lock:
        _prune_cancelled_runs(time.monotonic())
        if run_id in _cancelled_runs:
            return False
        _active_processes[run_id].add(proc)
        return True


async def _unregister_process(run_id: Optional[str], proc: asyncio.subprocess.Process) -> None:
    if not run_id:
        return
    async with _active_processes_lock:
        processes = _active_processes.get(run_id)
        if not processes:
            return
        processes.discard(proc)
        if not processes:
            _active_processes.pop(run_id, None)


async def _terminate_process(proc: asyncio.subprocess.Process) -> str:
    if proc.returncode is not None:
        return "already_exited"

    try:
        if os.name == "posix" and isinstance(proc.pid, int):
            os.killpg(proc.pid, signal.SIGTERM)
        else:
            proc.terminate()
    except ProcessLookupError:
        return "already_exited"

    try:
        await asyncio.wait_for(proc.wait(), timeout=settings.COMMAND_TERMINATE_GRACE_SECONDS)
        return "terminated"
    except asyncio.TimeoutError:
        try:
            if os.name == "posix" and isinstance(proc.pid, int):
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except ProcessLookupError:
            return "already_exited"
        await proc.wait()
        return "killed"


async def cancel_run_processes(run_id: str) -> dict[str, int]:
    """Terminate every active child process owned by a workflow run."""
    async with _active_processes_lock:
        _prune_cancelled_runs(time.monotonic())
        # The tombstone and process snapshot are protected by the same lock as
        # registration. A process that arrives after this point is rejected by
        # _register_process instead of escaping cancellation.
        _cancelled_runs[run_id] = time.monotonic()
        processes = list(_active_processes.get(run_id, set()))

    if not processes:
        return {"matched": 0, "terminated": 0, "killed": 0, "cancelled": 1}

    results = await asyncio.gather(
        *(_terminate_process(proc) for proc in processes), return_exceptions=True
    )
    terminated = sum(result == "terminated" for result in results)
    killed = sum(result == "killed" for result in results)
    return {
        "matched": len(processes),
        "terminated": terminated,
        "killed": killed,
        "cancelled": 1,
    }


def _result(
    *,
    output: str,
    error: bool,
    attempts: int,
    started_at: float,
    error_type: Optional[str] = None,
    retryable: bool = False,
    fallback_used: bool = False,
    ambiguous_outcome: bool = False,
    external_execution_started: bool = False,
) -> ToolOutput:
    result: ToolOutput = {
        "output": output,
        "error": error,
        "attempts": attempts,
        "duration_ms": int((time.monotonic() - started_at) * 1000),
        "retryable": retryable,
        "fallback_used": fallback_used,
    }
    if ambiguous_outcome:
        result["ambiguous_outcome"] = True
    if external_execution_started:
        result["external_execution_started"] = True
    if error_type:
        result["error_type"] = error_type
    return result


async def run_command(
    cmd: str,
    args: list[str],
    stdin: Optional[str] = None,
    *,
    timeout_seconds: Optional[float] = None,
    retry: Optional[bool] = None,
    fallback: Optional[tuple[str, list[str]]] = None,
) -> ToolOutput:
    """Execute a command with bounded runtime and safe retry semantics.

    Retries are inferred only for known read-only kubectl, Helm, and Argo
    commands. Mutation commands are never retried or given a fallback unless a
    caller explicitly restructures them as a verified higher-level workflow.
    """
    started_at = time.monotonic()
    retry_safe = _is_retry_safe(cmd, args) if retry is None else bool(retry)
    max_attempts = settings.COMMAND_MAX_RETRY_ATTEMPTS if retry_safe else 1
    timeout = timeout_seconds or (
        settings.COMMAND_READ_TIMEOUT_SECONDS
        if retry_safe
        else settings.COMMAND_MUTATION_TIMEOUT_SECONDS
    )
    run_id = _current_run_id()
    last_error_type = "command_failed"
    last_retryable = False
    last_message = "Command failed without diagnostic output."

    for attempt in range(1, max_attempts + 1):
        proc: Optional[asyncio.subprocess.Process] = None
        try:
            proc = await asyncio.create_subprocess_exec(
                cmd,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                start_new_session=os.name == "posix",
            )
            registered = await _register_process(run_id, proc)
            if not registered:
                await _terminate_process(proc)
                return _result(
                    output=(
                        "Command was cancelled because its workflow run had already "
                        "received a Stop request."
                    ),
                    error=True,
                    attempts=attempt,
                    started_at=started_at,
                    error_type="cancelled",
                    retryable=False,
                    ambiguous_outcome=not retry_safe,
                    external_execution_started=True,
                )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=stdin.encode() if stdin is not None else None),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                await _terminate_process(proc)
                last_error_type, last_retryable = _classify_error("", timed_out=True)
                last_message = f"Command exceeded timeout of {timeout:g}s"
            else:
                stdout_text = stdout.decode(errors="replace").strip()
                stderr_text = stderr.decode(errors="replace").strip()

                if proc.returncode == 0:
                    output_text = (
                        stdout_text
                        or stderr_text
                        or "The command was executed successfully, but no output was returned."
                    )
                    return _result(
                        output=output_text,
                        error=False,
                        attempts=attempt,
                        started_at=started_at,
                    )

                last_message = stderr_text or stdout_text or f"exit code {proc.returncode}"
                last_error_type, last_retryable = _classify_error(last_message)

        except asyncio.CancelledError:
            if proc is not None:
                await asyncio.shield(_terminate_process(proc))
            logger.info("Cancelled command run_id=%s command=%s", run_id, cmd)
            raise
        except FileNotFoundError as exc:
            last_error_type = "executable_not_found"
            last_retryable = False
            last_message = str(exc)
        except Exception as exc:
            last_message = str(exc)
            last_error_type, last_retryable = _classify_error(last_message)
        finally:
            if proc is not None:
                await _unregister_process(run_id, proc)

        if not (retry_safe and last_retryable and attempt < max_attempts):
            break

        delay = min(
            settings.COMMAND_RETRY_BASE_DELAY_SECONDS
            * (settings.COMMAND_RETRY_EXPONENTIAL_BASE ** (attempt - 1)),
            settings.COMMAND_RETRY_MAX_DELAY_SECONDS,
        )
        logger.warning(
            "Retrying command run_id=%s command=%s attempt=%s/%s error_type=%s delay=%ss",
            run_id,
            cmd,
            attempt + 1,
            max_attempts,
            last_error_type,
            delay,
        )
        await asyncio.sleep(delay)

    if fallback is not None and retry_safe:
        fallback_cmd, fallback_args = fallback
        fallback_result = await run_command(
            fallback_cmd,
            fallback_args,
            timeout_seconds=timeout,
            retry=False,
        )
        if not fallback_result["error"]:
            fallback_result["output"] = (
                f"Primary command failed ({last_error_type}: {last_message}). "
                f"Safe fallback evidence follows:\n{fallback_result['output']}"
            )
            fallback_result["fallback_used"] = True
            fallback_result["attempts"] = max_attempts + fallback_result.get("attempts", 1)
            fallback_result["duration_ms"] = int((time.monotonic() - started_at) * 1000)
            return fallback_result

    return _result(
        output=(
            f"Error executing command {cmd} with args {args} "
            f"[type={last_error_type}, attempts={attempt}, "
            f"retryable={str(last_retryable).lower()}]: {last_message}"
        ),
        error=True,
        attempts=attempt,
        started_at=started_at,
        error_type=last_error_type,
        retryable=last_retryable,
    )
