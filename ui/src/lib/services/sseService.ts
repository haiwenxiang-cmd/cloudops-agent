import { getAuthHeaders } from "@/lib/api";
import {
  Event,
  ReadyEvent,
  CompletedEvent,
  WorkflowCompleteEvent,
  TokenUsageEvent,
  TTFTEvent,
  ThinkingEvent,
  ThinkingCompleteEvent,
  ConversationTitleGeneratedEvent,
  MemoryContextLoadedEvent,
} from "@/types/events";
import { ChatMessage, ToolExecution, TokenUsage } from "@/types/chat";

export interface StreamRecoveryNotice {
  conversationId: string;
  runId?: string;
  reason?: string;
  error?: string;
}

export interface RunTerminalNotice {
  status: string;
  runId?: string;
  partialReasons: string[];
  durationMs?: number;
}

export interface ChatServiceCallbacks {
  onMessage?: (message: ChatMessage) => void;
  onToolExecuting?: (execution: ToolExecution) => void;
  onToolResult?: (execution: ToolExecution) => void;
  onToolsPending?: (executions: ToolExecution[]) => void;
  onToolAwaitingApproval?: (execution: ToolExecution) => void;
  onToolApproved?: (execution: ToolExecution) => void;
  onToolDenied?: (execution: ToolExecution) => void;
  onToolError?: (execution: ToolExecution) => void;
  onToolCancelled?: (execution: ToolExecution) => void;
  onToolProgress?: (
    execution: ToolExecution,
    message?: string,
    progress?: number,
  ) => void;
  onThinking?: (token: string, conversationId: string) => void;
  onThinkingComplete?: (content: string, durationMs: number) => void;
  onToken?: (token: string, conversationId: string) => void;
  onTokenUsage?: (usage: TokenUsage, source: "turn_check" | "main") => void;
  onTTFT?: (duration: number, runId: string) => void;
  onError?: (error: string) => void;
  onRecoveryRequired?: (notice: StreamRecoveryNotice) => void;
  onRunTerminal?: (notice: RunTerminalNotice) => void;
  onComplete?: (duration_ms?: number) => void;
  onReady?: (runId: string) => void;
  onConversationTitleGenerated?: (
    conversationId: string,
    title: string,
    timestamp: number,
  ) => void;
  onMemoryContextLoaded?: (
    runId: string,
    documents: MemoryContextLoadedEvent["documents"],
  ) => void;
}

export class ChatService {
  private abortController: AbortController | null = null;
  private callbacks: ChatServiceCallbacks = {};
  private isConnected: boolean = false;
  private toolExecutions = new Map<string, ToolExecution>();
  private hasCompleted: boolean = false;
  private currentRunId: string | null = null;
  private currentConversationId: string | null = null;
  private lastEventId: number = 0;
  private intentionalDisconnect: boolean = false;
  private readonly maxReconnectAttempts = 3;

  constructor(callbacks: ChatServiceCallbacks = {}) {
    this.callbacks = callbacks;
  }

  private activeRunStorageKey(conversationId: string): string {
    return `skyflo:active-run:${conversationId}`;
  }

  getPersistedRunId(conversationId: string): string | null {
    if (typeof window === "undefined") return null;
    try {
      const raw = window.sessionStorage.getItem(
        this.activeRunStorageKey(conversationId),
      );
      if (!raw) return null;
      const saved = JSON.parse(raw) as { runId?: unknown; savedAt?: unknown };
      const savedAt = Number(saved.savedAt);
      if (
        typeof saved.runId !== "string" ||
        !saved.runId ||
        !Number.isFinite(savedAt) ||
        Date.now() - savedAt > 15 * 60 * 1000
      ) {
        this.clearPersistedRun(conversationId);
        return null;
      }
      return saved.runId;
    } catch {
      this.clearPersistedRun(conversationId);
      return null;
    }
  }

  private persistActiveRun(): void {
    if (
      typeof window === "undefined" ||
      !this.currentRunId ||
      !this.currentConversationId
    ) {
      return;
    }
    window.sessionStorage.setItem(
      this.activeRunStorageKey(this.currentConversationId),
      JSON.stringify({ runId: this.currentRunId, savedAt: Date.now() }),
    );
  }

  clearPersistedRun(conversationId?: string): void {
    if (typeof window === "undefined") return;
    const target = conversationId || this.currentConversationId;
    if (target) {
      window.sessionStorage.removeItem(this.activeRunStorageKey(target));
    }
  }

  async startStream(
    messages: ChatMessage[],
    conversationId: string,
  ): Promise<void> {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL + "/agent/chat";
    let controller: AbortController | null = null;

    try {
      this.toolExecutions.clear();
      this.hasCompleted = false;

      this.disconnect();
      this.clearPersistedRun(conversationId);
      this.currentRunId = null;
      this.currentConversationId = conversationId;
      this.lastEventId = 0;
      this.intentionalDisconnect = false;
      controller = new AbortController();
      this.abortController = controller;

      const requestBody: any = {
        conversation_id: conversationId,
        messages: messages.map((msg) => ({
          role: msg.type === "user" ? "user" : "assistant",
          content: msg.content,
        })),
      };

      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          "Cache-Control": "no-cache",
          ...(await this.getAuthHeaders()),
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `HTTP ${response.status}: ${response.statusText} - ${errorText}`,
        );
      }

      if (!response.body) {
        throw new Error("No response body available");
      }

      this.isConnected = true;
      await this.consumeWithReconnect(
        response.body,
        conversationId,
        controller,
      );
    } catch (error) {
      if (this.isAbortError(error)) return;

      let errorMessage = "Unknown error";
      if (error instanceof Error) {
        if (error.message.includes("Failed to fetch")) {
          errorMessage = `Cannot connect to backend at ${apiUrl}. Please ensure:\n1. Backend is running on port 8080\n2. CORS is configured\n3. Network connectivity is available`;
        } else if (error.message.includes("404")) {
          errorMessage = `Endpoint not found. Please verify the /agent/chat endpoint exists on your backend`;
        } else if (error.message.includes("500")) {
          errorMessage = `Backend server error. Check backend logs for details`;
        } else {
          errorMessage = error.message;
        }
      }

      this.callbacks.onError?.(errorMessage);
      this.clearPersistedRun(conversationId);
    } finally {
      if (this.abortController === controller) {
        this.abortController = null;
      }
    }
  }

  async startApprovalStream(
    callId: string,
    approve: boolean,
    reason?: string,
    conversationId?: string,
  ): Promise<void> {
    const apiUrl =
      process.env.NEXT_PUBLIC_API_URL + `/agent/approvals/${callId}`;
    let controller: AbortController | null = null;

    try {
      this.hasCompleted = false;

      this.disconnect();
      if (conversationId) this.clearPersistedRun(conversationId);
      this.currentRunId = null;
      this.currentConversationId = conversationId || null;
      this.lastEventId = 0;
      this.intentionalDisconnect = false;
      controller = new AbortController();
      this.abortController = controller;

      const requestBody: any = {
        approve,
        reason,
        conversation_id: conversationId,
      };

      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          "Cache-Control": "no-cache",
          ...(await this.getAuthHeaders()),
        },
        body: JSON.stringify(requestBody),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(
          `HTTP ${response.status}: ${response.statusText} - ${errorText}`,
        );
      }

      if (!response.body) {
        throw new Error("No response body available from approval endpoint");
      }

      this.isConnected = true;
      await this.consumeWithReconnect(
        response.body,
        conversationId || "",
        controller,
      );
    } catch (error) {
      if (this.isAbortError(error)) return;

      let errorMessage = "Unknown error";
      if (error instanceof Error) {
        if (error.message.includes("Failed to fetch")) {
          errorMessage = `Cannot connect to backend at ${apiUrl}. Please ensure:\n1. Backend is running on port 8080\n2. CORS is configured\n3. Network connectivity is available`;
        } else if (error.message.includes("404")) {
          errorMessage = `Approval endpoint not found. Please verify the /approvals/${callId} endpoint exists on your backend`;
        } else if (error.message.includes("500")) {
          errorMessage = `Backend server error. Check backend logs for details`;
        } else {
          errorMessage = error.message;
        }
      }

      this.callbacks.onError?.(errorMessage);
      if (conversationId) this.clearPersistedRun(conversationId);
    } finally {
      if (this.abortController === controller) {
        this.abortController = null;
      }
    }
  }

  async resumePersistedStream(conversationId: string): Promise<boolean> {
    const runId = this.getPersistedRunId(conversationId);
    if (!runId) return false;

    const apiUrl =
      process.env.NEXT_PUBLIC_API_URL +
      `/agent/runs/${encodeURIComponent(runId)}/events`;
    this.disconnect();
    this.toolExecutions.clear();
    this.hasCompleted = false;
    this.currentRunId = runId;
    this.currentConversationId = conversationId;
    // A page refresh destroys rendered partial state, so replay from zero.
    // In-process network reconnects still resume from lastEventId below.
    this.lastEventId = 0;
    this.intentionalDisconnect = false;
    const controller = new AbortController();
    this.abortController = controller;

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          "Cache-Control": "no-cache",
          ...(await this.getAuthHeaders()),
        },
        body: JSON.stringify({ conversation_id: conversationId, last_event_id: 0 }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(
          `SSE refresh resume failed: HTTP ${response.status} ${response.statusText}`,
        );
      }
      this.isConnected = true;
      await this.consumeWithReconnect(response.body, conversationId, controller);
      return true;
    } catch (error) {
      if (this.isAbortError(error)) return false;
      this.clearPersistedRun(conversationId);
      this.callbacks.onError?.(
        error instanceof Error ? error.message : "Unable to resume active run",
      );
      return false;
    } finally {
      if (this.abortController === controller) this.abortController = null;
    }
  }

  private async consumeWithReconnect(
    initialBody: ReadableStream<Uint8Array>,
    conversationId: string,
    controller: AbortController,
  ): Promise<void> {
    let body = initialBody;
    let reconnectAttempt = 0;

    while (true) {
      await this.parseSSEStream(body);

      if (
        this.hasCompleted ||
        this.intentionalDisconnect ||
        controller.signal.aborted
      ) {
        return;
      }

      if (!this.currentRunId || !conversationId) {
        throw new Error("SSE disconnected before a resumable run was established");
      }

      const resumeUrl =
        process.env.NEXT_PUBLIC_API_URL +
        `/agent/runs/${encodeURIComponent(this.currentRunId)}/events`;
      let resumedBody: ReadableStream<Uint8Array> | null = null;
      let lastError: unknown = null;

      while (
        resumedBody === null &&
        reconnectAttempt < this.maxReconnectAttempts
      ) {
        reconnectAttempt += 1;
        await new Promise((resolve) =>
          setTimeout(resolve, 500 * 2 ** (reconnectAttempt - 1)),
        );
        if (controller.signal.aborted || this.intentionalDisconnect) return;

        try {
          const response = await fetch(resumeUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Accept: "text/event-stream",
              "Cache-Control": "no-cache",
              ...(await this.getAuthHeaders()),
            },
            body: JSON.stringify({
              conversation_id: conversationId,
              last_event_id: this.lastEventId,
            }),
            signal: controller.signal,
          });

          if (!response.ok || !response.body) {
            throw new Error(
              `SSE resume failed: HTTP ${response.status} ${response.statusText}`,
            );
          }
          resumedBody = response.body;
        } catch (error) {
          if (this.isAbortError(error)) return;
          lastError = error;
        }
      }

      if (resumedBody === null) {
        const detail = lastError instanceof Error ? `: ${lastError.message}` : "";
        throw new Error(
          `SSE reconnect failed after ${this.maxReconnectAttempts} attempts${detail}`,
        );
      }

      this.isConnected = true;
      reconnectAttempt = 0;
      body = resumedBody;
    }
  }

  private async parseSSEStream(
    body: ReadableStream<Uint8Array>,
  ): Promise<void> {
    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();

        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          this.processSSELine(line);
        }
      }
    } catch (error) {
      if (!this.isAbortError(error)) throw error;
    } finally {
      reader.releaseLock();
      this.isConnected = false;
    }
  }

  private processSSELine(line: string): void {
    if (line.startsWith("id: ")) {
      const parsed = Number.parseInt(line.substring(4).trim(), 10);
      if (Number.isFinite(parsed)) {
        this.lastEventId = Math.max(this.lastEventId, parsed);
      }
      return;
    }

    if (line.startsWith("event: ")) {
      return;
    }

    if (line.startsWith("data: ")) {
      const jsonData = line.substring(6);

      if (jsonData.trim() === "") return;

      try {
        const eventData = JSON.parse(jsonData);

        if (eventData.type === "stream.gap") {
          this.hasCompleted = true;
          this.clearPersistedRun();
          if (this.currentConversationId) {
            this.callbacks.onRecoveryRequired?.({
              conversationId: this.currentConversationId,
              runId:
                typeof eventData.run_id === "string" ? eventData.run_id : undefined,
              reason:
                typeof eventData.reason === "string" ? eventData.reason : undefined,
              error:
                typeof eventData.error === "string" ? eventData.error : undefined,
            });
          } else {
            this.callbacks.onError?.(
              String(
                eventData.error ||
                  "The live event history expired. Reload the conversation to restore its persisted state.",
              ),
            );
          }
        } else if (eventData.type) {
          this.handleSSEEvent(eventData as Event);
        } else if (eventData.status === "error" && eventData.error) {
          this.callbacks.onError?.(String(eventData.error));
        } else if (eventData.run_id && !eventData.type) {
          const readyEvent: ReadyEvent = {
            type: "ready",
            run_id: eventData.run_id,
          };
          this.handleSSEEvent(readyEvent);
        } else if (eventData.status === "completed" && eventData.result) {
          const workflowCompleteEvent: WorkflowCompleteEvent = {
            type: "workflow_complete",
            run_id: eventData.run_id,
            result: eventData.result,
            status: "completed",
          };
          this.handleSSEEvent(workflowCompleteEvent);
        } else if (eventData.status === "completed") {
          const completedEvent: CompletedEvent = {
            type: "completed",
            status: "completed",
            run_id: eventData.run_id,
          };
          this.handleSSEEvent(completedEvent);
        } else if (eventData.status === "awaiting_approval") {
          const workflowCompleteEvent: WorkflowCompleteEvent = {
            type: "workflow_complete",
            run_id: eventData.run_id,
            result: eventData.result,
            status: "awaiting_approval",
          };
          this.handleSSEEvent(workflowCompleteEvent);
        }
      } catch (error) {}
    }
  }

  private handleSSEEvent(event: Event): void {
    switch (event.type) {
      case "ready":
        this.currentRunId = event.run_id;
        this.persistActiveRun();
        this.callbacks.onReady?.(event.run_id);
        break;

      case "tool.executing":
        const executingTool: ToolExecution = {
          call_id: event.call_id,
          tool: event.tool,
          title: event.title,
          args: event.args,
          status: "executing",
          timestamp: event.timestamp,
        };
        this.toolExecutions.set(event.call_id, executingTool);
        this.callbacks.onToolExecuting?.(executingTool);
        break;

      case "tool.result":
        const existingExecution = this.toolExecutions.get(event.call_id);
        if (existingExecution) {
          const completedTool: ToolExecution = {
            ...existingExecution,
            status: "completed",
            result: event.result,
          };
          this.toolExecutions.set(event.call_id, completedTool);
          this.callbacks.onToolResult?.(completedTool);
        }
        break;

      case "tool.awaiting_approval":
        const awaitingApprovalTool: ToolExecution = {
          call_id: event.call_id,
          tool: event.tool,
          title: event.title,
          args: event.args,
          status: "awaiting_approval",
          timestamp: event.timestamp,
          requires_approval: true,
        };
        this.toolExecutions.set(event.call_id, awaitingApprovalTool);
        this.callbacks.onToolAwaitingApproval?.(awaitingApprovalTool);
        break;

      case "tool.approved":
        const approvedExecution = this.toolExecutions.get(event.call_id);
        if (approvedExecution) {
          const approvedTool: ToolExecution = {
            ...approvedExecution,
            status: "approved",
          };
          this.toolExecutions.set(event.call_id, approvedTool);
          this.callbacks.onToolApproved?.(approvedTool);
        } else {
          const approvedTool: ToolExecution = {
            call_id: event.call_id,
            tool: event.tool,
            title: event.title,
            args: event.args,
            status: "approved",
            timestamp: event.timestamp,
          };
          this.toolExecutions.set(event.call_id, approvedTool);
          this.callbacks.onToolApproved?.(approvedTool);
        }
        break;

      case "tool.denied":
        const deniedExecution = this.toolExecutions.get(event.call_id);
        if (deniedExecution) {
          const deniedTool: ToolExecution = {
            ...deniedExecution,
            status: "denied",
          };
          this.toolExecutions.set(event.call_id, deniedTool);
          this.callbacks.onToolDenied?.(deniedTool);
        } else {
          const deniedTool: ToolExecution = {
            call_id: event.call_id,
            tool: event.tool,
            title: event.title,
            args: event.args,
            status: "denied",
            timestamp: event.timestamp,
          };
          this.toolExecutions.set(event.call_id, deniedTool);
          this.callbacks.onToolDenied?.(deniedTool);
        }
        break;

      case "tool.error":
        const errorExecution = this.toolExecutions.get(event.call_id);
        if (errorExecution) {
          const errorTool: ToolExecution = {
            ...errorExecution,
            status: "error",
            error: event.error,
          };
          this.toolExecutions.set(event.call_id, errorTool);
          this.callbacks.onToolError?.(errorTool);
        } else {
          const newErrorTool: ToolExecution = {
            call_id: event.call_id,
            tool: event.tool,
            title: event.title,
            args: {},
            status: "error",
            timestamp: event.timestamp,
            error: event.error,
          };
          this.toolExecutions.set(event.call_id, newErrorTool);
          this.callbacks.onToolError?.(newErrorTool);
        }
        break;

      case "tool.cancelled": {
        const existingExecution = this.toolExecutions.get(event.call_id);
        const cancelledTool: ToolExecution = existingExecution
          ? {
              ...existingExecution,
              status: "cancelled",
              error: event.error || "Cancelled by user",
            }
          : {
              call_id: event.call_id,
              tool: event.tool,
              title: event.title,
              args: event.args || {},
              status: "cancelled",
              timestamp: event.timestamp,
              error: event.error || "Cancelled by user",
            };
        this.toolExecutions.set(event.call_id, cancelledTool);
        this.callbacks.onToolCancelled?.(cancelledTool);
        break;
      }

      case "tools.pending":
        const pendingList: ToolExecution[] = [];

        for (const t of event.tools || []) {
          const exec: ToolExecution = {
            call_id: t.call_id,
            tool: t.tool,
            title: t.title,
            args: t.args || {},
            status: "pending",
            timestamp: t.timestamp,
            requires_approval: t.requires_approval,
          };
          this.toolExecutions.set(t.call_id, exec);
          pendingList.push(exec);
        }
        if (pendingList.length > 0) {
          this.callbacks.onToolsPending?.(pendingList);
        }
        break;

      case "thinking": {
        if (this.hasCompleted) {
          return;
        }
        const thinkingEvent = event as ThinkingEvent;
        this.callbacks.onThinking?.(
          thinkingEvent.text,
          thinkingEvent.conversation_id,
        );
        break;
      }

      case "thinking.complete": {
        if (this.hasCompleted) {
          return;
        }
        const e = event as ThinkingCompleteEvent;
        this.callbacks.onThinkingComplete?.(e.content ?? "", e.duration_ms);
        break;
      }

      case "token":
        if (this.hasCompleted) {
          return;
        }
        this.callbacks.onToken?.(event.text, event.conversation_id);
        break;

      case "token.usage": {
        const usageEvent = event as TokenUsageEvent;
        this.callbacks.onTokenUsage?.(
          {
            prompt_tokens: usageEvent.prompt_tokens,
            completion_tokens: usageEvent.completion_tokens,
            total_tokens: usageEvent.total_tokens,
            cached_tokens: usageEvent.cached_tokens ?? 0,
          },
          usageEvent.source,
        );
        break;
      }

      case "ttft": {
        const ttftEvent = event as TTFTEvent;
        this.callbacks.onTTFT?.(ttftEvent.duration, ttftEvent.run_id);
        break;
      }

      case "error":
        this.callbacks.onError?.(event.error);
        break;

      case "workflow.error":
      case "workflow_error":
        this.callbacks.onError?.(event.error);
        break;

      case "completed":
        if (!this.hasCompleted) {
          this.hasCompleted = true;
          this.clearPersistedRun();
          const completedEvent = event as CompletedEvent;
          this.callbacks.onRunTerminal?.({
            status: completedEvent.status,
            runId: completedEvent.run_id,
            partialReasons: completedEvent.partial_reasons ?? [],
            durationMs: completedEvent.duration_ms,
          });
          this.callbacks.onComplete?.(completedEvent.duration_ms);
        }
        break;

      case "workflow_complete":
        if (!this.hasCompleted) {
          this.hasCompleted = true;
          this.clearPersistedRun();
          const completedEvent = event as WorkflowCompleteEvent;
          this.callbacks.onRunTerminal?.({
            status: completedEvent.status,
            runId: completedEvent.run_id,
            partialReasons: completedEvent.partial_reasons ?? [],
            durationMs: completedEvent.duration_ms,
          });
          this.callbacks.onComplete?.(completedEvent.duration_ms);
        }
        break;

      case "heartbeat":
        break;

      case "memory.context.loaded": {
        const memEvent = event as MemoryContextLoadedEvent;
        this.callbacks.onMemoryContextLoaded?.(memEvent.run_id, memEvent.documents);
        break;
      }

      case "memory.search":
      case "memory.write.created":
      case "memory.write.blocked":
      case "memory.policy.denied":
      case "memory.promotion.proposed":
      case "memory.safety.flagged":
        // These are informational events; no UI callback needed in Phase 1
        break;

      case "conversation.title.generated": {
        const titleEvent = event as ConversationTitleGeneratedEvent;
        this.callbacks.onConversationTitleGenerated?.(
          titleEvent.conversation_id,
          titleEvent.title,
          titleEvent.timestamp,
        );
        if (typeof window !== "undefined") {
          window.dispatchEvent(
            new CustomEvent("conversation:title-generated", {
              detail: {
                conversationId: titleEvent.conversation_id,
                title: titleEvent.title,
                timestamp: titleEvent.timestamp,
              },
            }),
          );
        }
        break;
      }

      default:
        break;
    }
  }

  private async getAuthHeaders(): Promise<Record<string, string>> {
    try {
      return await getAuthHeaders();
    } catch (error) {
      return {};
    }
  }

  private isAbortError(error: unknown): boolean {
    return (
      typeof error === "object" &&
      error !== null &&
      "name" in error &&
      error.name === "AbortError"
    );
  }

  disconnect(clearPersisted: boolean = false): void {
    this.intentionalDisconnect = true;
    if (clearPersisted) this.clearPersistedRun();
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    this.isConnected = false;
    this.hasCompleted = false;
  }

  isConnectedToStream(): boolean {
    return this.isConnected;
  }

  getToolExecutions(): ToolExecution[] {
    return Array.from(this.toolExecutions.values());
  }

  getToolExecution(callId: string): ToolExecution | undefined {
    return this.toolExecutions.get(callId);
  }
}
