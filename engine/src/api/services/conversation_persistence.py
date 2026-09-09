import json
import logging
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from tortoise.exceptions import DoesNotExist

from ..models.conversation import Conversation, Message, TokenUsageMetrics
from ..models.mutation import ExecutionStatus, MutationOperation, VerificationStatus

logger = logging.getLogger(__name__)

ACTIVE_TOOL_STATUSES = {"pending", "executing", "approved", "awaiting_approval"}
SAFE_VERIFICATION_REASONS = {
    "helm_manifest_invalid",
    "helm_manifest_unavailable",
    "helm_status_error",
    "invalid_helm_status_json",
    "operation_not_found",
    "partial_commit_detected",
    "verification_adapter_error",
    "verification_adapter_not_supported",
    "verification_already_in_progress",
    "verification_budget_exhausted",
    "verification_cancelled",
    "verification_lease_expired",
    "verification_timeout",
}


def project_mutation_operations(
    messages: List[Dict[str, Any]], operations: List[MutationOperation]
) -> List[Dict[str, Any]]:
    """Overlay durable mutation truth on the denormalized chat projection."""
    projected = deepcopy(messages)
    by_call_id = {str(op.call_id): op for op in operations if op.call_id}

    for message in projected:
        if message.get("type") != "assistant":
            continue
        segments = message.get("segments")
        if not isinstance(segments, list):
            continue
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("kind") != "tool":
                continue
            execution = segment.get("toolExecution")
            if not isinstance(execution, dict):
                continue
            call_id = str(execution.get("call_id") or segment.get("id") or "")
            operation = by_call_id.get(call_id)
            if operation is None:
                continue

            verification = operation.verification_status
            execution_status = operation.execution_status
            verification_result = getattr(operation, "verification_result", {})
            if not isinstance(verification_result, dict):
                verification_result = {}
            verification_reason = verification_result.get("reason")
            execution["operation_id"] = str(operation.id)
            execution["execution_status"] = execution_status.value
            execution["verification_status"] = verification.value
            # Expose only stable, non-sensitive recovery metadata. The complete
            # verification evidence remains in the journal and is intentionally
            # not copied into the chat projection.
            if (
                isinstance(verification_reason, str)
                and verification_reason in SAFE_VERIFICATION_REASONS
            ):
                execution["verification_reason"] = verification_reason
            else:
                execution.pop("verification_reason", None)
            execution["partial_commit_detected"] = bool(
                verification_result.get("partial_commit_detected")
                or verification_reason == "partial_commit_detected"
            )
            execution["requires_verification"] = verification in {
                VerificationStatus.PENDING,
                VerificationStatus.VERIFYING,
                VerificationStatus.INCONCLUSIVE,
            }

            if verification == VerificationStatus.PASSED:
                execution["status"] = "completed"
                execution.pop("error", None)
                execution["result"] = [
                    {"type": "text", "text": "Mutation completed and verification passed."}
                ]
            elif verification == VerificationStatus.NEEDS_REVIEW:
                execution["status"] = "error"
                execution["error"] = (
                    "Automatic verification reached its bounded limit. Manual review is required."
                )
            elif verification == VerificationStatus.FAILED:
                execution["status"] = "error"
                execution["error"] = "Mutation postcondition verification failed."
            elif execution_status == ExecutionStatus.CANCELLED:
                execution["status"] = "cancelled"
            elif execution_status == ExecutionStatus.FAILED:
                execution["status"] = "error"
                execution["error"] = "Mutation execution failed before an external request started."
            elif execution_status in {
                ExecutionStatus.EXECUTING,
                ExecutionStatus.REPORTED_SUCCESS,
                ExecutionStatus.UNKNOWN,
            }:
                execution["status"] = "executing"
                execution["result"] = [
                    {
                        "type": "text",
                        "text": (
                            "Mutation outcome is being reconciled from authoritative "
                            "infrastructure state."
                        ),
                    }
                ]
    return projected


def mark_active_tool_segments_cancelled(
    messages: List[Dict[str, Any]],
    run_id: Optional[str],
    reason: str,
) -> List[Dict[str, Any]]:
    """Mark active tools in the latest assistant turn as a terminal cancellation."""
    assistant = next(
        (message for message in reversed(messages) if message.get("type") == "assistant"),
        None,
    )
    if assistant is None:
        return []

    cancelled: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = assistant.get("segments", []) or []
    for segment in segments:
        if segment.get("kind") != "tool":
            continue
        execution = segment.get("toolExecution", {})
        if execution.get("status") not in ACTIVE_TOOL_STATUSES:
            continue
        execution_run_id = execution.get("run_id")
        if run_id and execution_run_id and str(execution_run_id) != str(run_id):
            continue
        execution["status"] = "cancelled"
        execution["error"] = reason
        execution["result"] = [{"type": "text", "text": reason}]
        segment["toolExecution"] = execution
        cancelled.append(dict(execution))

    assistant["segments"] = segments
    return cancelled


class ConversationPersistenceService:
    def __init__(self):
        self._usage_buffers: Dict[str, Dict[str, Any]] = {}
        self._message: Message | None = None

    async def project_authoritative_mutation_statuses(
        self, conversation_id: str, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        operations = await MutationOperation.filter(conversation_id=conversation_id).all()
        return project_mutation_operations(messages, operations)

    def _clear_message(self) -> None:
        self._message = None

    def _usage_key(self, conversation_id: Optional[str], run_id: Optional[str]) -> Optional[str]:
        if not conversation_id or not run_id:
            return None
        return f"{str(conversation_id)}:{str(run_id)}"

    def _get_usage_buffer(
        self, conversation_id: Optional[str], run_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        key = self._usage_key(conversation_id, run_id)
        if not key:
            return None
        if key not in self._usage_buffers:
            self._usage_buffers[key] = {
                "model": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "cost": 0.0,
                "ttft_ms": None,
                "ttr_ms": None,
            }
        return self._usage_buffers.get(key)

    async def _get_next_sequence(self, conversation) -> int:
        latest_message = (
            await Message.filter(conversation_id=conversation.id).order_by("-sequence").first()
        )
        sequence: int = latest_message.sequence + 1 if latest_message else 1
        return sequence

    def record_token_usage(
        self,
        conversation_id: Optional[str],
        run_id: Optional[str],
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cached_tokens: Optional[int] = None,
        cost: float = 0.0,
        model: Optional[str] = None,
    ) -> None:
        buffer = self._get_usage_buffer(conversation_id, run_id)
        if not buffer:
            return
        if model:
            buffer["model"] = model
        buffer["prompt_tokens"] += max(prompt_tokens or 0, 0)
        buffer["completion_tokens"] += max(completion_tokens or 0, 0)
        buffer["total_tokens"] += max(total_tokens or 0, 0)
        if cached_tokens is not None:
            buffer["cached_tokens"] += max(cached_tokens or 0, 0)
        buffer["cost"] += max(cost or 0.0, 0.0)

    def record_ttft(
        self, conversation_id: Optional[str], run_id: Optional[str], duration_ms: Optional[int]
    ) -> None:
        buffer = self._get_usage_buffer(conversation_id, run_id)
        if not buffer:
            return
        if duration_ms is not None:
            buffer["ttft_ms"] = duration_ms

    def record_ttr(
        self, conversation_id: Optional[str], run_id: Optional[str], duration_ms: Optional[int]
    ) -> None:
        buffer = self._get_usage_buffer(conversation_id, run_id)
        if not buffer:
            return
        if duration_ms is not None:
            buffer["ttr_ms"] = duration_ms

    def _snapshot_usage(
        self, conversation_id: Optional[str], run_id: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        key = self._usage_key(conversation_id, run_id)
        if not key:
            return None

        buffer = self._usage_buffers.get(key)
        if not buffer:
            return None

        return {
            "model": buffer.get("model"),
            "prompt_tokens": buffer.get("prompt_tokens", 0),
            "completion_tokens": buffer.get("completion_tokens", 0),
            "total_tokens": buffer.get("total_tokens", 0),
            "cached_tokens": buffer.get("cached_tokens", 0),
            "cost": buffer.get("cost", 0.0),
            "ttft_ms": buffer.get("ttft_ms"),
            "ttr_ms": buffer.get("ttr_ms"),
        }

    def _clear_usage_buffer(self, conversation_id: Optional[str], run_id: Optional[str]) -> None:
        key = self._usage_key(conversation_id, run_id)
        if key and key in self._usage_buffers:
            self._usage_buffers.pop(key, None)

    async def _apply_usage_to_latest_assistant(
        self, conversation_id: str, run_id: Optional[str], finalize: bool
    ) -> None:
        usage_snapshot = self._snapshot_usage(conversation_id, run_id)
        if not usage_snapshot:
            if finalize:
                self._clear_usage_buffer(conversation_id, run_id)
            return

        try:
            conversation = await Conversation.get(id=conversation_id)
        except Exception:
            if finalize:
                self._clear_usage_buffer(conversation_id, run_id)
            return

        messages: List[Dict[str, Any]] = conversation.messages_json or []
        if not messages:
            if finalize:
                self._clear_usage_buffer(conversation_id, run_id)
            return

        last_message = messages[-1]
        if last_message.get("type") != "assistant":
            if finalize:
                self._clear_usage_buffer(conversation_id, run_id)
            return

        metrics = TokenUsageMetrics(**usage_snapshot)
        usage_dict = metrics.model_dump()
        last_message["token_usage"] = usage_dict
        await conversation.update_from_dict({"messages_json": messages}).save()

        msg_id = last_message.get("id")
        msg_row: Message | None = None
        if msg_id:
            try:
                msg_row = await Message.get(id=uuid.UUID(str(msg_id)))
            except Exception as exc:
                logger.debug(
                    "Could not update token_usage on Message row %s for conversation %s: %s",
                    msg_id,
                    conversation_id,
                    exc,
                    exc_info=True,
                )

        if not msg_row:
            msg_row = (
                await Message.filter(conversation_id=conversation_id, role="assistant")
                .order_by("-sequence")
                .first()
            )
            if msg_row:
                logger.debug(
                    "Falling back to latest assistant message for token_usage "
                    "update; msg_id=%s conversation_id=%s resolved_message_id=%s",
                    msg_id,
                    conversation_id,
                    msg_row.id,
                )

        if msg_row:
            msg_row.token_usage = usage_dict
            await msg_row.save()

        if finalize:
            self._clear_usage_buffer(conversation_id, run_id)

    async def apply_usage_snapshot(self, conversation_id: str, run_id: Optional[str]) -> None:
        await self._apply_usage_to_latest_assistant(conversation_id, run_id, finalize=False)

    async def finalize_usage_snapshot(self, conversation_id: str, run_id: Optional[str]) -> None:
        await self._apply_usage_to_latest_assistant(conversation_id, run_id, finalize=True)
        self._clear_message()

    async def append_user_message(self, conversation_id: str, content: str, timestamp: int) -> None:
        conversation = await Conversation.get(id=conversation_id)
        messages: List[Dict[str, Any]] = conversation.messages_json or []

        if (
            messages
            and messages[-1].get("type") == "user"
            and messages[-1].get("content") == content
        ):
            return

        user_message = {
            "id": str(uuid.uuid4()),
            "type": "user",
            "content": content,
            "timestamp": timestamp,
        }

        messages.append(user_message)
        await conversation.update_from_dict({"messages_json": messages}).save()
        created_at = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
        sequence: int = await self._get_next_sequence(conversation=conversation)
        await Message.create(
            conversation_id=conversation.id,
            role="user",
            content=content,
            sequence=sequence,
            created_at=created_at,
        )

    async def append_text_segment(
        self, conversation_id: str, text: str, timestamp: int, run_id: Optional[str] = None
    ) -> None:
        conversation = await Conversation.get(id=conversation_id)
        messages: List[Dict[str, Any]] = conversation.messages_json or []

        if not messages or messages[-1].get("type") != "assistant":
            assistant_message_id = str(uuid.uuid4())
            assistant_message = {
                "id": assistant_message_id,
                "type": "assistant",
                "content": text,
                "timestamp": timestamp,
                "segments": [
                    {
                        "kind": "text",
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "timestamp": timestamp,
                    }
                ],
            }
            messages.append(assistant_message)

            created_at = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)
            sequence = await self._get_next_sequence(conversation=conversation)
            token_usage = None
            if run_id:
                usage_snapshot = self._snapshot_usage(conversation_id, run_id)
                if usage_snapshot:
                    token_usage = TokenUsageMetrics(**usage_snapshot).model_dump()

            self._message = await Message.create(
                id=uuid.UUID(assistant_message_id),
                conversation=conversation,
                role="assistant",
                content=text,
                sequence=sequence,
                message_metadata={"segments": assistant_message["segments"]},
                token_usage=token_usage,
                created_at=created_at,
            )

        else:
            assistant = messages[-1]
            segments: List[Dict[str, Any]] = assistant.get("segments", [])

            segments.append(
                {
                    "kind": "text",
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "timestamp": timestamp,
                }
            )

            assistant["content"] = assistant.get("content", "") + text
            assistant["segments"] = segments

            if self._message:
                self._message.content = self._message.content + text
                self._message.message_metadata = {"segments": segments}
                await self._message.save()

        if run_id:
            usage_snapshot = self._snapshot_usage(conversation_id, run_id)
            if usage_snapshot:
                assistant = messages[-1]
                assistant["token_usage"] = TokenUsageMetrics(**usage_snapshot).model_dump()
                if self._message:
                    self._message.token_usage = assistant["token_usage"]
                    await self._message.save()

        await conversation.update_from_dict({"messages_json": messages}).save()

    async def append_thinking_segment(
        self,
        conversation_id: str,
        text: str,
        timestamp: int,
        duration_ms: int = 0,
        run_id: Optional[str] = None,
    ) -> None:
        conversation = await Conversation.get(id=conversation_id)
        messages: List[Dict[str, Any]] = conversation.messages_json or []
        created_at = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)

        if not messages or messages[-1].get("type") != "assistant":
            msg_id = str(uuid.uuid4())
            assistant_message = {
                "id": msg_id,
                "type": "assistant",
                "content": "",
                "timestamp": timestamp,
                "segments": [
                    {
                        "kind": "thinking",
                        "id": str(uuid.uuid4()),
                        "text": text,
                        "isComplete": True,
                        "durationMs": duration_ms,
                        "timestamp": timestamp,
                    }
                ],
            }
            messages.append(assistant_message)

            seq = await self._get_next_sequence(conversation)
            self._message = await Message.create(
                id=uuid.UUID(msg_id),
                conversation=conversation,
                role="assistant",
                content="",
                sequence=seq,
                message_metadata={"segments": assistant_message["segments"]},
                created_at=created_at,
            )
        else:
            assistant = messages[-1]
            segments: List[Dict[str, Any]] = assistant.get("segments", [])

            segments.append(
                {
                    "kind": "thinking",
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "isComplete": True,
                    "durationMs": duration_ms,
                    "timestamp": timestamp,
                }
            )

            assistant["segments"] = segments

            assistant_id = assistant.get("id")
            if assistant_id:
                try:
                    msg_row = await Message.get(id=uuid.UUID(assistant_id))
                    msg_row.message_metadata = {"segments": segments}
                    await msg_row.save()
                    self._message = msg_row
                except DoesNotExist:
                    seq = await self._get_next_sequence(conversation)
                    self._message = await Message.create(
                        id=uuid.UUID(assistant_id),
                        conversation=conversation,
                        role="assistant",
                        content="",
                        sequence=seq,
                        message_metadata={"segments": segments},
                        created_at=created_at,
                    )
                except Exception:
                    logger.exception(
                        "Failed to persist thinking segment for message %s in conversation %s",
                        assistant_id,
                        conversation_id,
                    )

        if run_id:
            usage_snapshot = self._snapshot_usage(conversation_id, run_id)
            if usage_snapshot:
                assistant = messages[-1]
                assistant["token_usage"] = TokenUsageMetrics(**usage_snapshot).model_dump()
                if self._message:
                    self._message.token_usage = assistant["token_usage"]
                    await self._message.save()

        await conversation.update_from_dict({"messages_json": messages}).save()

    async def append_tool_segment(
        self,
        conversation_id: str,
        tool_execution: Dict[str, Any],
        timestamp: int,
        run_id: Optional[str] = None,
    ) -> None:
        conversation = await Conversation.get(id=conversation_id)
        messages: List[Dict[str, Any]] = conversation.messages_json or []
        created_at = datetime.fromtimestamp(timestamp / 1000.0, tz=timezone.utc)

        if not messages or messages[-1].get("type") != "assistant":
            assistant_id = str(uuid.uuid4())
            assistant_message = {
                "id": assistant_id,
                "type": "assistant",
                "content": "",
                "timestamp": timestamp,
                "segments": [],
            }
            messages.append(assistant_message)

            seq = await self._get_next_sequence(conversation)
            self._message = await Message.create(
                id=uuid.UUID(assistant_id),
                conversation=conversation,
                role="assistant",
                content="",
                sequence=seq,
                message_metadata={"segments": []},
                created_at=created_at,
            )

        assistant = messages[-1]
        segments: List[Dict[str, Any]] = assistant.get("segments", [])
        call_id = tool_execution.get("call_id")

        existing_index = next(
            (
                i
                for i in range(len(segments))
                if segments[i].get("kind") == "tool" and segments[i].get("id") == call_id
            ),
            -1,
        )

        if existing_index >= 0:
            return

        segment = {
            "kind": "tool",
            "id": call_id,
            "toolExecution": tool_execution,
            "timestamp": timestamp,
        }

        segments.append(segment)
        assistant["segments"] = segments

        assistant_id = assistant.get("id")
        if assistant_id:
            try:
                msg_row = await Message.get(id=uuid.UUID(str(assistant_id)))
                msg_row.message_metadata = {"segments": segments}
                await msg_row.save()
                self._message = msg_row
            except DoesNotExist:
                seq = await self._get_next_sequence(conversation)
                self._message = await Message.create(
                    id=uuid.UUID(str(assistant_id)),
                    conversation=conversation,
                    role="assistant",
                    content=str(assistant.get("content", "")),
                    sequence=seq,
                    message_metadata={"segments": segments},
                    created_at=created_at,
                )
            except Exception:
                logger.exception(
                    "Failed to persist tool segment for message %s in conversation %s",
                    assistant_id,
                    conversation_id,
                )

        if run_id:
            usage_snapshot = self._snapshot_usage(conversation_id, run_id)
            if usage_snapshot:
                assistant["token_usage"] = TokenUsageMetrics(**usage_snapshot).model_dump()
                if self._message:
                    self._message.token_usage = assistant["token_usage"]
                    await self._message.save()

        await conversation.update_from_dict({"messages_json": messages}).save()

    async def update_tool_segment_status(
        self,
        conversation_id: str,
        call_id: str,
        status: str,
        error: Optional[str] = None,
        result: Optional[List[Dict[str, Any]]] = None,
        allow_cancelled_transition: bool = False,
    ) -> None:
        conversation = await Conversation.get(id=conversation_id)
        messages: List[Dict[str, Any]] = conversation.messages_json or []
        if not messages:
            return

        for assistant in reversed(messages):
            if assistant.get("type") != "assistant":
                continue
            segments: List[Dict[str, Any]] = assistant.get("segments", []) or []
            for i in range(len(segments) - 1, -1, -1):
                seg = segments[i]
                if seg.get("kind") != "tool" or seg.get("id") != call_id:
                    continue
                exec_obj = seg.get("toolExecution", {})
                if (
                    exec_obj.get("status") == "cancelled"
                    and status != "cancelled"
                    and not allow_cancelled_transition
                ):
                    return
                exec_obj["status"] = status
                if error is not None:
                    exec_obj["error"] = error
                if result is not None:
                    exec_obj["result"] = result
                seg["toolExecution"] = exec_obj
                segments[i] = seg
                assistant["segments"] = segments
                await conversation.update_from_dict({"messages_json": messages}).save()
                assistant_id = assistant.get("id")
                if assistant_id:
                    try:
                        msg_row = await Message.get(id=uuid.UUID(str(assistant_id)))
                        msg_row.message_metadata = {"segments": segments}
                        await msg_row.save()
                        self._message = msg_row
                    except Exception:
                        logger.debug(
                            "Could not update tool status metadata on Message row %s", assistant_id
                        )
                return

    async def cancel_active_tool_segments(
        self,
        conversation_id: str,
        run_id: Optional[str],
        reason: str = "Cancelled by user",
    ) -> List[Dict[str, Any]]:
        conversation = await Conversation.get(id=conversation_id)
        messages: List[Dict[str, Any]] = conversation.messages_json or []
        cancelled = mark_active_tool_segments_cancelled(messages, run_id, reason)
        if not cancelled:
            return []

        await conversation.update_from_dict({"messages_json": messages}).save()
        assistant = next(
            (message for message in reversed(messages) if message.get("type") == "assistant"),
            None,
        )
        assistant_id = assistant.get("id") if assistant else None
        if assistant_id:
            try:
                msg_row = await Message.get(id=uuid.UUID(str(assistant_id)))
                msg_row.message_metadata = {"segments": assistant.get("segments", [])}
                await msg_row.save()
                self._message = msg_row
            except Exception:
                logger.exception(
                    "Failed to persist cancelled tool metadata for message %s",
                    assistant_id,
                )
        return cancelled

    async def build_llm_messages(self, conversation: Conversation) -> List[Dict[str, Any]]:
        messages_json: List[Dict[str, Any]] = conversation.messages_json or []

        llm_messages: List[Dict[str, Any]] = []

        for msg in messages_json:
            mtype = msg.get("type")
            if mtype == "user":
                content = str(msg.get("content", ""))
                if content:
                    llm_messages.append({"role": "user", "content": content})
                continue

            if mtype != "assistant":
                continue

            segments: List[Dict[str, Any]] = [
                s for s in (msg.get("segments", []) or []) if s.get("kind") != "thinking"
            ]
            first_tool_index = next(
                (i for i, segment in enumerate(segments) if segment.get("kind") == "tool"), -1
            )
            last_tool_index = max(
                (i for i, segment in enumerate(segments) if segment.get("kind") == "tool"),
                default=-1,
            )

            if first_tool_index > 0:
                pre_text_parts: List[str] = []
                for segment in segments[:first_tool_index]:
                    if segment.get("kind") == "text":
                        pre_text_parts.append(str(segment.get("text", "")))
                pre_text = "".join(pre_text_parts)
                if pre_text:
                    llm_messages.append({"role": "assistant", "content": pre_text})

            tool_calls: List[Dict[str, Any]] = []
            tool_segments: List[Dict[str, Any]] = []
            for segment in segments:
                if segment.get("kind") != "tool":
                    continue
                tool_exec = segment.get("toolExecution", {}) or {}
                tool_name = tool_exec.get("tool") or ""
                call_id = str(tool_exec.get("call_id") or "").strip()
                args_obj = tool_exec.get("args") or {}
                args_str = json.dumps(args_obj) if isinstance(args_obj, dict) else str(args_obj)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args_str},
                    }
                )
                tool_segments.append(segment)

            if tool_calls:
                llm_messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})

                for segment in tool_segments:
                    tool_exec = segment.get("toolExecution", {}) or {}
                    tool_name = tool_exec.get("tool") or ""
                    call_id = str(tool_exec.get("call_id") or "").strip()
                    result_blocks = tool_exec.get("result") or []
                    result_content = ""
                    if result_blocks:
                        for block in result_blocks:
                            if isinstance(block, dict) and block.get("type") == "text":
                                result_content += str(block.get("text", ""))
                            else:
                                result_content += str(block)
                    else:
                        status = (tool_exec.get("status") or "").lower()
                        if status == "awaiting_approval":
                            result_content = "Pending tool approval from the user"
                        elif status == "denied":
                            result_content = "Tool call was denied by the user"
                        elif status == "error":
                            err = tool_exec.get("error") or "Tool execution failed"
                            result_content = str(err)

                    llm_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call_id,
                            "name": tool_name,
                            "content": result_content,
                        }
                    )

            if last_tool_index >= 0 and last_tool_index < len(segments) - 1:
                post_text_parts: List[str] = []
                for segment in segments[last_tool_index + 1 :]:
                    if segment.get("kind") == "text":
                        post_text_parts.append(str(segment.get("text", "")))
                post_text = "".join(post_text_parts)
                if post_text:
                    llm_messages.append({"role": "assistant", "content": post_text})

            if not tool_calls:
                content = str(msg.get("content", ""))
                if content:
                    llm_messages.append({"role": "assistant", "content": content})

        return llm_messages

    async def build_llm_messages_for_title_generation(
        self, conversation: Conversation
    ) -> List[Dict[str, Any]]:
        """
        Build a simplified message list for title generation.
        - Includes only 'user' and 'assistant' messages with plain text content.
        - Excludes any tool call metadata and 'tool' role messages.
        """
        messages_json: List[Dict[str, Any]] = conversation.messages_json or []

        simplified: List[Dict[str, Any]] = []

        for msg in messages_json:
            mtype = msg.get("type")
            if mtype == "user":
                content = str(msg.get("content", ""))
                if content:
                    simplified.append({"role": "user", "content": content})
                continue

            if mtype != "assistant":
                continue

            # Prefer concatenated text content if present; otherwise, stitch text segments only
            content = str(msg.get("content", ""))
            if content:
                simplified.append({"role": "assistant", "content": content})
                continue

            segments: List[Dict[str, Any]] = msg.get("segments", []) or []
            text_parts: List[str] = []
            for segment in segments:
                if segment.get("kind") == "text":
                    text_parts.append(str(segment.get("text", "")))
            stitched = "".join(text_parts).strip()
            if stitched:
                simplified.append({"role": "assistant", "content": stitched})

        return simplified

    async def set_title(self, conversation_id: str, title: str) -> bool:
        conversation = await Conversation.get(id=conversation_id)
        current = (conversation.title or "").strip() if conversation.title is not None else ""
        if current:
            return False
        normalized = (title or "").strip()
        if not normalized:
            return False
        await conversation.update_from_dict({"title": normalized}).save()
        return True
