"""Type definitions for MCP tools."""

from typing_extensions import NotRequired, TypedDict


class ToolOutput(TypedDict):
    """Structured output from tool commands."""

    output: str
    error: bool
    error_type: NotRequired[str]
    retryable: NotRequired[bool]
    attempts: NotRequired[int]
    duration_ms: NotRequired[int]
    fallback_used: NotRequired[bool]
    ambiguous_outcome: NotRequired[bool]
    external_execution_started: NotRequired[bool]
