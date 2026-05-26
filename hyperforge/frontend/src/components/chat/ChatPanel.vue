<template>
  <div class="chat-outer">
    <!-- ── Main chat panel ─────────────────────────────────────────── -->
    <div class="chat-panel">
      <div class="chat-header">
        <span class="chat-title">Test Chat</span>
        <div class="chat-meta">
          <span class="session-label">session: {{ chatStore.sessionId.slice(0, 8) }}…</span>
          <button class="icon-btn" title="New session" @click="chatStore.reset()">↺</button>
          <button
            class="icon-btn debug-toggle"
            :class="{ active: showDebug }"
            :title="showDebug ? 'Hide WS frames' : 'Show WS frames'"
            @click="showDebug = !showDebug"
          >
            ⚡<span v-if="chatStore.wsFrames.length" class="frame-badge">{{ chatStore.wsFrames.length }}</span>
          </button>
        </div>
      </div>

      <div ref="messagesEl" class="messages">
        <div v-if="chatStore.messages.length === 0" class="empty-chat">
          <div class="empty-icon">💬</div>
          <p>Ask anything to test the pipeline</p>
        </div>

        <div
          v-for="msg in chatStore.messages"
          :key="msg.id"
          class="message"
          :class="msg.role"
        >
          <!-- User bubble -->
          <div v-if="msg.role === 'user'" class="bubble user-bubble">
            {{ msg.text }}
          </div>

          <!-- Assistant bubble -->
          <div v-else class="bubble assistant-bubble">
            <!-- Steps taken -->
            <div v-if="msg.steps.length > 0" class="steps">
              <details>
                <summary>{{ msg.steps.length }} step(s) executed</summary>
                <ul>
                  <li v-for="(step, i) in msg.steps" :key="i">
                    <code>{{ step.module }}</code> — {{ step.title || '' }}
                  </li>
                </ul>
              </details>
            </div>

            <!-- Contexts retrieved -->
            <div v-if="msg.contexts.length > 0" class="contexts">
              <details>
                <summary>{{ msg.contexts.length }} context source(s)</summary>
                <div v-for="(ctx, i) in msg.contexts" :key="i" class="context-source">
                  <div class="context-source-label">{{ ctx.source }}</div>
                  <div v-for="(chunk, j) in ctx.chunks.slice(0, 3)" :key="j" class="chunk-text">
                    {{ chunk.text.slice(0, 300) }}{{ chunk.text.length > 300 ? '…' : '' }}
                  </div>
                </div>
              </details>
            </div>

            <!-- Answer -->
            <div v-if="msg.text" class="answer-text">{{ msg.text }}</div>
            <div v-else-if="msg.streaming" class="streaming-indicator">
              <span class="dot" /><span class="dot" /><span class="dot" />
            </div>

            <!-- OAuth authentication request -->
            <a
              v-if="msg.pendingOAuthUrl"
              class="oauth-btn"
              :href="msg.pendingOAuthUrl"
              target="_blank"
              rel="noopener noreferrer"
            >
              🔐 Authenticate with provider
            </a>

            <!-- Error -->
            <div v-if="msg.error" class="error-text">Error: {{ msg.error }}</div>
          </div>
        </div>
      </div>

      <div class="input-area">
        <div v-if="!workflowStore.activeAgentId" class="no-agent">
          Load a config first to start chatting.
        </div>
        <div v-else class="input-row">
          <textarea
            ref="inputEl"
            v-model="question"
            class="chat-input"
            placeholder="Ask a question…"
            rows="2"
            :disabled="chatStore.streaming"
            @keydown.enter.exact.prevent="submit"
          />
          <button
            class="send-btn"
            :disabled="chatStore.streaming || !question.trim()"
            @click="submit"
          >
            {{ chatStore.streaming ? '…' : '→' }}
          </button>
        </div>
      </div>
    </div>

    <!-- ── WS debug side panel ─────────────────────────────────────── -->
    <div v-if="showDebug" class="debug-panel">
      <div class="debug-header">
        <span class="debug-title">WS Frames</span>
        <button class="icon-btn" title="Clear frames" @click="chatStore.wsFrames.length = 0">✕ Clear</button>
      </div>
      <div ref="debugEl" class="debug-frames">
        <div v-if="chatStore.wsFrames.length === 0" class="debug-empty">
          No frames yet — send a message to start.
        </div>
        <div
          v-for="(frame, i) in chatStore.wsFrames"
          :key="i"
          class="frame"
          :class="frameClass(frame.operation)"
        >
          <div class="frame-meta">
            <span class="frame-ts">{{ frame.ts }}</span>
            <span class="frame-op">{{ opLabel(frame.operation) }}</span>
          </div>
          <pre class="frame-body">{{ frame.raw }}</pre>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch, nextTick } from 'vue'
import { useChatStore } from '@/stores/chat'
import { useWorkflowStore } from '@/stores/workflow'
import { AnswerOperation } from '@/types/arag'

const chatStore = useChatStore()
const workflowStore = useWorkflowStore()

const question = ref('')
const messagesEl = ref<HTMLElement | null>(null)
const inputEl = ref<HTMLTextAreaElement | null>(null)
const debugEl = ref<HTMLElement | null>(null)
const showDebug = ref(false)

async function submit() {
  const q = question.value.trim()
  if (!q || chatStore.streaming || !workflowStore.activeAgentId) return
  question.value = ''
  await chatStore.send(workflowStore.activeAgentId, q, workflowStore.activeWorkflowId)
}

// Scroll chat to bottom when messages change
watch(
  () => chatStore.messages.length,
  async () => {
    await nextTick()
    if (messagesEl.value) {
      messagesEl.value.scrollTop = messagesEl.value.scrollHeight
    }
  },
)

// Auto-scroll debug panel to bottom on new frames
watch(
  () => chatStore.wsFrames.length,
  async () => {
    await nextTick()
    if (debugEl.value) {
      debugEl.value.scrollTop = debugEl.value.scrollHeight
    }
  },
)

const OP_LABELS: Record<number, string> = {
  [AnswerOperation.ANSWER]: 'ANSWER',
  [AnswerOperation.START]: 'START',
  [AnswerOperation.DONE]: 'DONE',
  [AnswerOperation.ERROR]: 'ERROR',
  [AnswerOperation.AGENT_REQUEST]: 'AGENT_REQUEST',
}

function opLabel(op?: number): string {
  if (op === undefined) return '?'
  return OP_LABELS[op] ?? `op:${op}`
}

function frameClass(op?: number): string {
  if (op === AnswerOperation.START) return 'op-start'
  if (op === AnswerOperation.ANSWER) return 'op-answer'
  if (op === AnswerOperation.DONE) return 'op-done'
  if (op === AnswerOperation.ERROR) return 'op-error'
  if (op === AnswerOperation.AGENT_REQUEST) return 'op-agent-request'
  return 'op-other'
}
</script>

<style scoped>
/* ── Outer wrapper ────────────────────────────────────────────────── */
.chat-outer {
  display: flex;
  height: 100%;
  overflow: hidden;
}

/* ── Main chat panel ──────────────────────────────────────────────── */
.chat-panel {
  display: flex;
  flex-direction: column;
  flex: 1;
  min-width: 0;
  background: #13131f;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-bottom: 1px solid #222;
  flex-shrink: 0;
}

.chat-title {
  font-size: 13px;
  font-weight: 700;
  color: #bbb;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.chat-meta {
  display: flex;
  align-items: center;
  gap: 8px;
}

.session-label {
  font-size: 11px;
  color: #555;
  font-family: monospace;
}

.icon-btn {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
  padding: 0;
  transition: color 0.1s;
}

.icon-btn:hover {
  color: #ccc;
}

.debug-toggle {
  position: relative;
}

.debug-toggle.active {
  color: #f5a623;
}

.frame-badge {
  position: absolute;
  top: -6px;
  right: -8px;
  background: #f5a623;
  color: #000;
  font-size: 9px;
  font-weight: 700;
  border-radius: 8px;
  padding: 1px 4px;
  line-height: 1.4;
  pointer-events: none;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px 12px;
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.empty-chat {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: #555;
  gap: 8px;
  margin-top: 40px;
}

.empty-icon {
  font-size: 40px;
  opacity: 0.4;
}

.empty-chat p {
  font-size: 13px;
}

.message {
  display: flex;
}

.message.user {
  justify-content: flex-end;
}

.message.assistant {
  justify-content: flex-start;
}

.bubble {
  max-width: 88%;
  border-radius: 12px;
  padding: 10px 14px;
  font-size: 13px;
  line-height: 1.6;
}

.user-bubble {
  background: #7c5cfc;
  color: #fff;
  border-bottom-right-radius: 3px;
}

.assistant-bubble {
  background: #1e1e30;
  color: #ddd;
  border-bottom-left-radius: 3px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  width: 100%;
}

.steps,
.contexts {
  font-size: 12px;
  color: #888;
}

.steps details,
.contexts details {
  cursor: pointer;
}

.steps summary,
.contexts summary {
  color: #7c5cfc;
  font-weight: 600;
  padding: 2px 0;
  outline: none;
}

.steps ul {
  margin: 6px 0 0 16px;
  padding: 0;
  list-style: disc;
}

.steps li {
  margin: 2px 0;
}

.context-source {
  margin: 8px 0 0;
  border-left: 2px solid #333;
  padding-left: 10px;
}

.context-source-label {
  font-size: 11px;
  font-weight: 700;
  color: #2196f3;
  margin-bottom: 4px;
}

.chunk-text {
  font-size: 11px;
  color: #777;
  margin: 3px 0;
  line-height: 1.5;
}

.answer-text {
  color: #e0e0e0;
  white-space: pre-wrap;
}

.streaming-indicator {
  display: flex;
  gap: 5px;
  align-items: center;
  height: 20px;
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #7c5cfc;
  animation: pulse 1.2s ease-in-out infinite;
}

.dot:nth-child(2) {
  animation-delay: 0.2s;
}

.dot:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes pulse {
  0%, 100% { opacity: 0.3; transform: scale(0.85); }
  50% { opacity: 1; transform: scale(1); }
}

.error-text {
  color: #ff5555;
  font-size: 12px;
}

.oauth-btn {
  align-self: flex-start;
  display: inline-block;
  background: #2a2a40;
  color: #7c5cfc;
  border: 1px solid #7c5cfc;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
  transition: background 0.15s;
}

.oauth-btn:hover {
  background: #3a3a55;
}

.input-area {
  border-top: 1px solid #222;
  padding: 10px 12px;
  flex-shrink: 0;
}

.no-agent {
  font-size: 12px;
  color: #555;
  text-align: center;
  padding: 8px;
}

.input-row {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.chat-input {
  flex: 1;
  background: #1a1a2e;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 8px 10px;
  color: #e0e0e0;
  font-size: 13px;
  resize: none;
  outline: none;
  font-family: inherit;
  line-height: 1.5;
  transition: border-color 0.15s;
}

.chat-input:focus {
  border-color: #7c5cfc;
}

.chat-input:disabled {
  opacity: 0.5;
}

.send-btn {
  background: #7c5cfc;
  border: none;
  color: #fff;
  width: 40px;
  height: 40px;
  border-radius: 8px;
  font-size: 18px;
  cursor: pointer;
  flex-shrink: 0;
  transition: background 0.15s, opacity 0.15s;
}

.send-btn:hover:not(:disabled) {
  background: #6a4de0;
}

.send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

/* ── Debug side panel ─────────────────────────────────────────────── */
.debug-panel {
  width: 360px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: #0d0d1a;
  border-left: 1px solid #2a2a3a;
  overflow: hidden;
}

.debug-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-bottom: 1px solid #2a2a3a;
  flex-shrink: 0;
}

.debug-title {
  font-size: 11px;
  font-weight: 700;
  color: #f5a623;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

.debug-header .icon-btn {
  font-size: 11px;
  color: #555;
}

.debug-frames {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.debug-empty {
  font-size: 11px;
  color: #444;
  text-align: center;
  margin-top: 24px;
}

.frame {
  border-radius: 6px;
  border-left: 3px solid #333;
  background: #13131f;
  overflow: hidden;
}

.frame-meta {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 8px;
  border-bottom: 1px solid #1e1e2e;
}

.frame-ts {
  font-size: 10px;
  color: #555;
  font-family: monospace;
}

.frame-op {
  font-size: 10px;
  font-weight: 700;
  font-family: monospace;
}

.frame-body {
  margin: 0;
  padding: 6px 8px;
  font-size: 10px;
  font-family: monospace;
  color: #aaa;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 200px;
  overflow-y: auto;
}

/* Operation colour coding */
.op-start  { border-left-color: #2196f3; }
.op-start .frame-op  { color: #2196f3; }

.op-answer { border-left-color: #4caf50; }
.op-answer .frame-op { color: #4caf50; }

.op-done   { border-left-color: #00bcd4; }
.op-done .frame-op   { color: #00bcd4; }

.op-error  { border-left-color: #ff5555; }
.op-error .frame-op  { color: #ff5555; }

.op-agent-request { border-left-color: #f5a623; }
.op-agent-request .frame-op { color: #f5a623; }

.op-other  { border-left-color: #555; }
.op-other .frame-op  { color: #888; }
</style>
