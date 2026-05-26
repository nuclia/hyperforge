import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'
import { v4 as uuidv4 } from 'uuid'
import { connectChat } from '@/api/chatWs'
import type { ChatSocket } from '@/api/chatWs'
import { AnswerOperation } from '@/types/arag'
import type { AragAnswer, AragContext, AragStep, Feedback } from '@/types/arag'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  contexts: AragContext[]
  steps: AragStep[]
  citations?: Record<string, unknown>
  streaming: boolean
  error?: string
  /** Set when the agent is waiting for user input (AGENT_REQUEST). */
  pendingFeedback?: Feedback
  /**
   * Set when the agent emitted an OAuth URL via AGENT_REQUEST. Rendered as a
   * user-clickable link so popup blockers don't interfere (the auto-open
   * approach is unreliable since this fires from a WS message, not a gesture).
   */
  pendingOAuthUrl?: string
}

export interface WsFrame {
  ts: string
  operation?: number
  raw: string
}

export const useChatStore = defineStore('chat', () => {
  const messages = ref<ChatMessage[]>([])
  const streaming = ref(false)
  const connected = ref(false)
  const sessionId = ref('ephemeral')
  const wsFrames = ref<WsFrame[]>([])

  // Active WebSocket — kept across sends (keep_open=true)
  const socket = shallowRef<ChatSocket | null>(null)

  // Current assistant message being assembled (reset on new send)
  let currentAssistant: ChatMessage | null = null

  function _ensureSocket(agentId: string, workflowId: string) {
    if (socket.value?.isOpen()) return

    socket.value = connectChat(
      agentId,
      workflowId,
      sessionId.value,
      // onMessage
      (msg: AragAnswer) => {
        wsFrames.value.push({
          ts: new Date().toISOString().slice(11, 23),
          operation: msg.operation,
          raw: JSON.stringify(msg, null, 2),
        })
        if (!currentAssistant) return
        _applyChunk(currentAssistant, msg)
        if (
          msg.operation === AnswerOperation.DONE ||
          msg.operation === AnswerOperation.ERROR
        ) {
          currentAssistant.streaming = false
          streaming.value = false
          currentAssistant = null
        }
      },
      // onError
      (err: string) => {
        if (currentAssistant) {
          currentAssistant.error = err
          currentAssistant.streaming = false
          currentAssistant = null
        }
        streaming.value = false
        connected.value = false
      },
      // onClose
      () => {
        connected.value = false
        if (currentAssistant) {
          currentAssistant.streaming = false
          currentAssistant = null
        }
        streaming.value = false
      },
    )
    connected.value = true
  }

  function reset(agentId?: string, workflowId?: string) {
    socket.value?.close()
    socket.value = null
    messages.value = []
    currentAssistant = null
    streaming.value = false
    connected.value = false
    // Re-open connection immediately so it's ready
    if (agentId && workflowId) {
      _ensureSocket(agentId, workflowId)
    }
  }

  function send(agentId: string, question: string, workflowId: string = 'default') {
    if (streaming.value) return

    _ensureSocket(agentId, workflowId)

    messages.value.push({
      id: uuidv4(),
      role: 'user',
      text: question,
      contexts: [],
      steps: [],
      streaming: false,
    })

    const assistantMsg: ChatMessage = {
      id: uuidv4(),
      role: 'assistant',
      text: '',
      contexts: [],
      steps: [],
      streaming: true,
    }
    messages.value.push(assistantMsg)
    // Read back the reactive proxy Vue created when it processed the push.
    // Mutating the original plain object (assistantMsg) bypasses Vue's
    // reactivity system, so the template never re-renders.
    currentAssistant = messages.value[messages.value.length - 1]
    streaming.value = true

    socket.value!.sendQuestion(question)
  }

  /** Called by the UI when the user submits a response to an AGENT_REQUEST */
  function submitFeedback(requestId: string, response: string) {
    if (!socket.value?.isOpen()) return
    // Clear the pendingFeedback on the last assistant message
    const last = [...messages.value].reverse().find((m: ChatMessage) => m.pendingFeedback?.request_id === requestId)
    if (last) {
      delete last.pendingFeedback
      // Re-enable streaming so we can receive the continued answer
      last.streaming = true
      currentAssistant = last
      streaming.value = true
    }
    socket.value.sendFeedback(requestId, response)
  }

  return {
    messages,
    streaming,
    connected,
    sessionId,
    wsFrames,
    reset,
    send,
    submitFeedback,
  }
})

function _applyChunk(msg: ChatMessage, chunk: AragAnswer) {
  if (chunk.step) {
    msg.steps.push(chunk.step)
  }
  if (chunk.context) {
    msg.contexts.push(chunk.context)
  }
  if (chunk.answer !== undefined) {
    msg.text += chunk.answer
    if (chunk.answer_citations) msg.citations = chunk.answer_citations
  }
  if (chunk.operation === AnswerOperation.AGENT_REQUEST) {
    if (chunk.feedback) {
      // Agent needs user input — pause streaming and surface the feedback form.
      msg.streaming = false
      msg.pendingFeedback = chunk.feedback
    }
    if (chunk.oauth?.oauth_url) {
      // Surface as a button rather than auto-opening: window.open() called
      // from a WS message handler is reliably blocked by browsers since
      // there's no preceding user gesture. The agent resumes on its own
      // once OAuth completes, so we don't pause streaming here.
      msg.pendingOAuthUrl = chunk.oauth.oauth_url
    }
  }
  if (chunk.operation === AnswerOperation.ERROR && chunk.exception) {
    msg.error = chunk.exception.detail
  }
}
