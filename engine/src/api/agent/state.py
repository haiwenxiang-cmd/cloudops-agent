import operator
import time
import uuid
from typing import Annotated, Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentState(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    messages: Annotated[List[Dict[str, Any]], operator.add] = Field(default_factory=list)
    pending_tools: List[Dict[str, Any]] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.time)
    end_time: float = 0.0
    duration: float = 0.0
    done: bool = False
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    environment: str = "development"
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    awaiting_approval: bool = False
    suppress_pending_event: bool = False
    ttft_emitted: bool = False
    approval_decisions: Dict[str, bool] = Field(default_factory=dict)
    approval_reasons: Dict[str, str] = Field(default_factory=dict)
    mutation_operations: List[Dict[str, Any]] = Field(default_factory=list)
    verification_required: bool = False
    verification_blocked: bool = False
    verification_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    loaded_toolsets: Dict[str, bool] = Field(
        default_factory=lambda: {"k8s": False, "memory": False}
    )

    memory_context_loaded: bool = False
    memory_hits: List[Dict[str, Any]] = Field(default_factory=list)
    memory_context_msg: Optional[Dict[str, Any]] = None

    class Config:
        arbitrary_types_allowed = True
