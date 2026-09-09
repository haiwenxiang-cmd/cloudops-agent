export type ReliabilityTone = "info" | "progress" | "success" | "warning" | "danger";

export interface ReliabilityPresentation {
  code: string;
  label: string;
  description: string;
  tone: ReliabilityTone;
}

export interface OperationReliabilityState {
  execution_status?: string;
  verification_status?: string;
  verification_reason?: string;
  partial_commit_detected?: boolean;
  requires_verification?: boolean;
}

export interface RunTerminalState {
  status?: string;
  partial_reasons?: string[];
}

const normalized = (value?: string): string => value?.trim().toLowerCase() ?? "";

/**
 * Convert durable operation state into user-facing recovery semantics.
 * This function deliberately does not infer from human-readable error text.
 */
export function mapOperationReliabilityState(
  state: OperationReliabilityState,
): ReliabilityPresentation | null {
  const execution = normalized(state.execution_status);
  const verification = normalized(state.verification_status);
  const reason = normalized(state.verification_reason);

  if (state.partial_commit_detected || reason === "partial_commit_detected") {
    return {
      code: "partial_commit_detected",
      label: "Partially applied — manual review required",
      description:
        "Some targets match this operation while others do not. Skyflo will not replay the entire mutation automatically.",
      tone: "danger",
    };
  }

  if (verification === "needs_review") {
    return {
      code: "needs_review",
      label: "Needs manual review",
      description:
        "Automatic verification reached its bounded limit. Inspect the infrastructure evidence before retrying the mutation.",
      tone: "warning",
    };
  }

  if (verification === "failed") {
    return {
      code: reason || "verification_failed",
      label: "Verification failed",
      description:
        "The observed infrastructure state does not satisfy the expected postcondition. The mutation may be absent, partial, or conflicting.",
      tone: "danger",
    };
  }

  if (verification === "passed") {
    const recovered = execution === "unknown";
    return {
      code: recovered ? "recovered_and_verified" : "verified",
      label: recovered ? "Applied — verified after recovery" : "Applied and verified",
      description: recovered
        ? "The original tool response was uncertain, but read-only reconciliation confirmed the expected infrastructure state."
        : "Read-only postcondition checks confirmed the expected infrastructure state.",
      tone: "success",
    };
  }

  if (verification === "inconclusive") {
    return {
      code: reason || "verification_inconclusive",
      label: "Evidence inconclusive",
      description:
        "The current evidence is insufficient. Skyflo will continue read-only verification within the configured recovery budget.",
      tone: "warning",
    };
  }

  if (verification === "pending" || verification === "verifying") {
    const outcomeUnknown = execution === "unknown";
    return {
      code: outcomeUnknown ? "unknown_verifying" : "verification_in_progress",
      label: outcomeUnknown ? "Outcome unknown — verifying" : "Verifying outcome",
      description:
        "Skyflo is checking authoritative infrastructure state with read-only operations. Do not retry this mutation yet.",
      tone: "progress",
    };
  }

  if (execution === "unknown") {
    return {
      code: "unknown",
      label: "Outcome unknown",
      description:
        "Skyflo could not prove whether the external side effect committed. Review the operation before retrying it.",
      tone: "warning",
    };
  }

  return null;
}

export function mapRunTerminalState(
  state: RunTerminalState,
): ReliabilityPresentation | null {
  if (normalized(state.status) !== "stop_partial") return null;

  return {
    code: "stop_partial",
    label: "Agent stopped; remote outcome not fully confirmed",
    description:
      "The local workflow stopped, but at least one remote cancellation or journal update could not be confirmed. Check unresolved operations before retrying a write.",
    tone: "warning",
  };
}

export function mapStreamRecoveryState(
  reason?: string,
): ReliabilityPresentation {
  if (normalized(reason) === "run_mapping_expired") {
    return {
      code: "run_mapping_expired",
      label: "Live update history expired",
      description:
        "Skyflo is reloading the authoritative persisted conversation. This transport gap does not by itself mean the infrastructure operation failed.",
      tone: "info",
    };
  }

  return {
    code: normalized(reason) || "stream_gap",
    label: "Live updates interrupted",
    description:
      "Skyflo is reloading the authoritative persisted conversation to recover the latest durable state.",
    tone: "info",
  };
}
