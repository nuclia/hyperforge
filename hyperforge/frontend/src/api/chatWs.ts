/**
 * WebSocket chat client for ARAG.
 *
 * Opens a persistent connection to the agent's WebSocket endpoint with
 * keep_open=true, allowing multiple questions per session.
 *
 * Protocol (mirrors `hyperforge/api/v1/interaction.py`):
 *   → InteractionRequest         { question, headers, arguments, operation: 0 }
 *   ← AragAnswer (stream)        { operation, answer?, exception?, feedback?, oauth?, ... }
 *   ↔ AGENT_REQUEST → client replies with UserToAgentInteraction.
 */

import {AnswerOperation, InteractionOperation} from "@/types/arag";
import type {AragAnswer, UserToAgentInteraction} from "@/types/arag";

export type MessageHandler = (msg: AragAnswer) => void;
export type ErrorHandler = (err: string) => void;
export type CloseHandler = () => void;
export type RawFrameHandler = (raw: string) => void;

export interface ChatSocket {
  sendQuestion(question: string, args?: Record<string, string>): void;
  sendFeedback(requestId: string, response: string): void;
  close(): void;
  isOpen(): boolean;
}

function wsUrl(agentId: string, workflowId: string, sessionId: string): string {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host;
  return (
    `${proto}://${host}/api/v1/agent/${encodeURIComponent(agentId)}` +
    `/workflow/${encodeURIComponent(workflowId)}` +
    `/session/${encodeURIComponent(sessionId)}/ws?keep_open=true`
  );
}

export function connectChat(
  agentId: string,
  workflowId: string,
  sessionId: string,
  onMessage: MessageHandler,
  onError: ErrorHandler,
  onClose: CloseHandler,
): ChatSocket {
  const url = wsUrl(agentId, workflowId, sessionId);
  const ws = new WebSocket(url);

  function handleText(text: string) {
    try {
      const msg = JSON.parse(text) as AragAnswer;
      if (msg.operation === undefined) {
        onMessage({
          ...msg,
          operation: AnswerOperation.ANSWER,
          answer: msg.answer ?? text,
        });
        return;
      }
      onMessage(msg);
    } catch {
      onError("Failed to parse server message");
    }
  }

  ws.onmessage = (event: MessageEvent) => {
    const data: unknown = event.data;
    if (typeof data === "string") {
      handleText(data);
    } else if (data instanceof Blob) {
      // Server may occasionally frame as binary — coerce to text.
      data
        .text()
        .then(handleText)
        .catch(() => onError("Failed to read binary frame"));
    } else if (data instanceof ArrayBuffer) {
      handleText(new TextDecoder().decode(data));
    } else {
      onError("Unexpected WebSocket frame type");
    }
  };

  ws.onerror = () => {
    onError("WebSocket connection error");
  };

  ws.onclose = () => {
    onClose();
  };

  function waitOpen(): Promise<void> {
    if (ws.readyState === WebSocket.OPEN) return Promise.resolve();
    return new Promise((resolve, reject) => {
      ws.onopen = () => resolve();
      const prev = ws.onerror;
      ws.onerror = (e) => {
        if (prev) (prev as EventListener)(e);
        reject(new Error("WebSocket failed to open"));
      };
    });
  }

  return {
    sendQuestion(question: string, args: Record<string, string> = {}) {
      // `headers` is required server-side (defaults to {}).
      const payload = JSON.stringify({
        question,
        headers: {},
        arguments: args,
        operation: InteractionOperation.QUESTION,
      });
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(payload);
      } else {
        waitOpen()
          .then(() => ws.send(payload))
          .catch((e) => onError(String(e)));
      }
    },

    sendFeedback(requestId: string, response: string) {
      const payload: UserToAgentInteraction = {
        op: "user_response",
        request_id: requestId,
        response,
      };
      ws.send(JSON.stringify(payload));
    },

    close() {
      if (
        ws.readyState === WebSocket.OPEN ||
        ws.readyState === WebSocket.CONNECTING
      ) {
        ws.close();
      }
    },

    isOpen() {
      return ws.readyState === WebSocket.OPEN;
    },
  };
}
