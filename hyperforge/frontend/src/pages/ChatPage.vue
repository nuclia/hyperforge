<template>
  <div class="chat-page">
    <!-- Header bar -->
    <header class="chat-header">
      <div class="header-left">
        <span class="chat-brand">Chat</span>
        <select class="agent-select" :value="activeAgentId" @change="onAgentChange">
          <option v-for="id in agentIds" :key="id" :value="id">
            {{ agentLabel(id) }}
          </option>
        </select>
        <select class="workflow-select" :value="activeWorkflowId" @change="onWorkflowChange">
          <option v-for="id in workflowIds" :key="id" :value="id">
            {{ workflowLabel(id) }}
          </option>
        </select>
      </div>
      <div class="header-right">
        <span :class="['conn-badge', connected ? 'conn-badge--on' : 'conn-badge--off']">
          {{ connected ? 'connected' : 'disconnected' }}
        </span>
        <button class="btn-icon" title="New session" @click="onReset">↺ New session</button>
      </div>
    </header>

    <!-- Main: messages + steps panel -->
    <div class="chat-body">
      <!-- Messages column -->
      <div class="messages-col">
        <div ref="messagesEl" class="messages-scroll">
          <div v-if="messages.length === 0" class="empty-state">
            <div class="empty-icon">💬</div>
            <p>Ask anything to test the pipeline</p>
            <p class="empty-sub">Using workflow: <strong>{{ activeWorkflowId }}</strong></p>
          </div>

          <TransitionGroup name="msg" tag="div" class="messages-list">
            <div v-for="msg in messages" :key="msg.id" :class="['message', msg.role]">

              <!-- User bubble -->
              <div v-if="msg.role === 'user'" class="bubble user-bubble">
                <div class="bubble-text">{{ msg.text }}</div>
              </div>

              <!-- Assistant bubble -->
              <div v-else class="bubble assistant-bubble">
                <!-- Thinking / steps -->
                <div v-if="msg.steps.length > 0" class="msg-steps">
                  <details open>
                    <summary class="steps-summary">
                      <span class="steps-icon">⚙</span>
                      {{ msg.steps.length }} step{{ msg.steps.length > 1 ? 's' : '' }}
                    </summary>
                    <div class="steps-list">
                      <div v-for="(step, i) in msg.steps" :key="i" class="step-item">
                        <span class="step-idx">{{ i + 1 }}</span>
                        <span class="step-module">{{ step.module }}</span>
                        <span v-if="step.title" class="step-title">{{ step.title }}</span>
                      </div>
                    </div>
                  </details>
                </div>

                <!-- Context sources -->
                <div v-if="msg.contexts.length > 0" class="msg-contexts">
                  <details>
                    <summary class="ctx-summary">
                      <span class="ctx-icon">📄</span>
                      {{ msg.contexts.length }} context source{{ msg.contexts.length > 1 ? 's' : '' }}
                    </summary>
                    <div class="ctx-list">
                      <div v-for="(ctx, ci) in msg.contexts" :key="ci" class="ctx-source">
                        <div class="ctx-source-name">{{ ctx.source }}</div>
                        <div v-for="(chunk, chi) in ctx.chunks.slice(0, 3)" :key="chi" class="ctx-chunk">
                          {{ chunk.text.slice(0, 400) }}{{ chunk.text.length > 400 ? '…' : '' }}
                        </div>
                        <div v-if="ctx.chunks.length > 3" class="ctx-more">
                          +{{ ctx.chunks.length - 3 }} more chunks
                        </div>
                      </div>
                    </div>
                  </details>
                </div>

                <!-- Streaming cursor -->
                <div v-if="msg.streaming" class="streaming-indicator">
                  <span class="dot" /><span class="dot" /><span class="dot" />
                </div>

                <!-- Answer text -->
                <div v-if="msg.text" class="bubble-text answer-text">
                  <pre class="answer-pre">{{ msg.text }}</pre>
                </div>

                <!-- Error -->
                <div v-if="msg.error" class="bubble-error">
                  ⚠ {{ msg.error }}
                </div>

                <!-- AGENT_REQUEST feedback form -->
                <div v-if="msg.pendingFeedback" class="feedback-block">
                  <div class="feedback-header">
                    <span class="feedback-icon">🤔</span>
                    <strong>Agent needs input</strong>
                    <span class="feedback-module">{{ msg.pendingFeedback.module }}</span>
                  </div>
                  <p class="feedback-question">{{ msg.pendingFeedback.question }}</p>
                  <FeedbackForm
                    :feedback="msg.pendingFeedback"
                    @submit="(r) => chatStore.submitFeedback(msg.pendingFeedback!.request_id, r)"
                  />
                </div>
              </div>
            </div>
          </TransitionGroup>
        </div>
      </div>

      <!-- Steps / reasoning panel (right sidebar) -->
      <div :class="['steps-panel', showSteps ? 'steps-panel--open' : '']">
        <div class="steps-panel-header">
          <span class="steps-panel-title">Reasoning trace</span>
          <button class="steps-panel-close" @click="showSteps = false">✕</button>
        </div>
        <div class="steps-panel-body">
          <div v-if="allSteps.length === 0" class="steps-empty">No steps yet</div>
          <div v-for="(entry, i) in allSteps" :key="i" class="trace-entry">
            <div class="trace-entry-header">
              <span class="trace-q-label">Q{{ entry.questionIndex + 1 }}</span>
              <span class="trace-step-idx">Step {{ entry.stepIndex + 1 }}</span>
              <span class="trace-module">{{ entry.step.module }}</span>
            </div>
            <div v-if="entry.step.title" class="trace-title">{{ entry.step.title }}</div>
          </div>

          <div v-if="allContexts.length > 0" class="trace-section-title">Retrieved contexts</div>
          <div v-for="(entry, i) in allContexts" :key="'ctx' + i" class="trace-ctx-entry">
            <div class="trace-ctx-source">{{ entry.source }}</div>
            <div class="trace-ctx-count">{{ entry.chunks }} chunks</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Input bar -->
    <div class="input-bar">
      <button
        :class="['trace-toggle', showSteps ? 'trace-toggle--active' : '']"
        title="Toggle reasoning trace"
        @click="showSteps = !showSteps"
      >⚙ Trace</button>

      <div class="input-wrap">
        <textarea
          ref="inputEl"
          v-model="inputText"
          class="question-input"
          placeholder="Ask a question…"
          rows="1"
          :disabled="chatStore.streaming"
          @keydown.enter.exact.prevent="onSend"
          @input="autoResize"
        />
      </div>

      <button
        class="send-btn"
        :disabled="!inputText.trim() || chatStore.streaming"
        @click="onSend"
      >
        <span v-if="chatStore.streaming" class="sending-dots">
          <span class="dot" /><span class="dot" /><span class="dot" />
        </span>
        <span v-else>Send ↑</span>
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { useWorkflowStore } from '@/stores/workflow'
import { useChatStore } from '@/stores/chat'
import FeedbackForm from '@/components/chat/FeedbackForm.vue'

const workflowStore = useWorkflowStore()
const chatStore = useChatStore()

// Agent / workflow selection (independent from editor page selection)
const activeAgentId = ref(workflowStore.activeAgentId ?? '')
const activeWorkflowId = ref(workflowStore.activeWorkflowId ?? 'default')

const agentIds = computed(() => workflowStore.agentIds)
const workflowIds = computed(() => {
  if (!activeAgentId.value) return ['default']
  return Object.keys(workflowStore.config[activeAgentId.value]?.workflows ?? {})
})

function agentLabel(id: string) {
  return workflowStore.config[id]?.title || id
}
function workflowLabel(id: string) {
  return workflowStore.config[activeAgentId.value]?.workflows[id]?.name || id
}

function onAgentChange(e: Event) {
  activeAgentId.value = (e.target as HTMLSelectElement).value
  activeWorkflowId.value = workflowIds.value[0] ?? 'default'
  chatStore.reset(activeAgentId.value, activeWorkflowId.value)
}
function onWorkflowChange(e: Event) {
  activeWorkflowId.value = (e.target as HTMLSelectElement).value
  chatStore.reset(activeAgentId.value, activeWorkflowId.value)
}

watch(
  () => workflowStore.activeAgentId,
  (id) => { if (id && !activeAgentId.value) activeAgentId.value = id },
  { immediate: true },
)

// Chat
const inputText = ref('')
const inputEl = ref<HTMLTextAreaElement | null>(null)
const messagesEl = ref<HTMLElement | null>(null)
const showSteps = ref(false)
const { messages, streaming, connected } = chatStore

function onSend() {
  const q = inputText.value.trim()
  if (!q || streaming) return
  inputText.value = ''
  nextTick(() => {
    if (inputEl.value) inputEl.value.style.height = 'auto'
  })
  chatStore.send(activeAgentId.value, q, activeWorkflowId.value)
}

function onReset() {
  chatStore.reset(activeAgentId.value, activeWorkflowId.value)
}

function autoResize(e: Event) {
  const el = e.target as HTMLTextAreaElement
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 200) + 'px'
}

// Auto-scroll to bottom on new messages
watch(
  () => messages.length,
  () => nextTick(() => {
    if (messagesEl.value) {
      messagesEl.value.scrollTop = messagesEl.value.scrollHeight
    }
  }),
)

// Aggregate all steps + contexts for the trace panel
const allSteps = computed(() => {
  const result: { questionIndex: number; stepIndex: number; step: { module: string; title?: string } }[] = []
  let qi = 0
  for (const msg of messages) {
    if (msg.role === 'assistant') {
      msg.steps.forEach((s, si) => result.push({ questionIndex: qi - 1, stepIndex: si, step: s }))
      qi++
    } else {
      qi++
    }
  }
  return result
})

const allContexts = computed(() => {
  const seen = new Map<string, number>()
  for (const msg of messages) {
    if (msg.role === 'assistant') {
      for (const ctx of msg.contexts) {
        seen.set(ctx.source, (seen.get(ctx.source) ?? 0) + ctx.chunks.length)
      }
    }
  }
  return Array.from(seen.entries()).map(([source, chunks]) => ({ source, chunks }))
})
</script>

<style scoped>
.chat-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
  background: #13131f;
}

/* ── Header ── */
.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px;
  height: 50px;
  background: #0f0f1c;
  border-bottom: 1px solid #1e1e30;
  flex-shrink: 0;
  gap: 12px;
}
.header-left, .header-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.chat-brand {
  font-size: 15px;
  font-weight: 800;
  color: #2196f3;
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.agent-select, .workflow-select {
  background: #1a1a2e;
  border: 1px solid #333;
  color: #ddd;
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 13px;
  cursor: pointer;
  outline: none;
  max-width: 200px;
}
.conn-badge {
  font-size: 11px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 99px;
}
.conn-badge--on { background: rgba(76,175,80,0.15); color: #4caf50; }
.conn-badge--off { background: rgba(255,85,85,0.15); color: #ff5555; }
.btn-icon {
  background: transparent;
  border: 1px solid #333;
  color: #bbb;
  padding: 5px 12px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-icon:hover { background: #1a1a2e; color: #fff; }

/* ── Body ── */
.chat-body {
  flex: 1;
  display: flex;
  overflow: hidden;
  position: relative;
}

/* ── Messages column ── */
.messages-col {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
.messages-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  flex: 1;
  color: #555;
  gap: 8px;
  min-height: 300px;
}
.empty-icon { font-size: 48px; opacity: 0.4; }
.empty-state p { font-size: 14px; margin: 0; }
.empty-sub { font-size: 12px; color: #444; }

.messages-list { display: flex; flex-direction: column; gap: 16px; }

.message { display: flex; }
.message.user { justify-content: flex-end; }
.message.assistant { justify-content: flex-start; }

.bubble {
  max-width: 72%;
  border-radius: 12px;
  padding: 12px 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.user-bubble {
  background: #1a1a2e;
  border: 1px solid #2a2a50;
  border-bottom-right-radius: 4px;
}
.assistant-bubble {
  background: #0f1220;
  border: 1px solid #1e1e30;
  border-bottom-left-radius: 4px;
  max-width: 85%;
}

.bubble-text { font-size: 14px; line-height: 1.6; color: #e0e0e0; }
.answer-text { white-space: pre-wrap; }
.answer-pre {
  white-space: pre-wrap;
  word-break: break-word;
  font-family: inherit;
  font-size: 14px;
  line-height: 1.7;
  margin: 0;
  color: #e0e0e0;
}

/* Steps summary */
.msg-steps details, .msg-contexts details {
  background: #0a0a14;
  border: 1px solid #1e1e30;
  border-radius: 8px;
  padding: 8px 12px;
}
.steps-summary, .ctx-summary {
  cursor: pointer;
  font-size: 12px;
  font-weight: 600;
  color: #888;
  list-style: none;
  display: flex;
  align-items: center;
  gap: 6px;
}
.steps-summary::-webkit-details-marker,
.ctx-summary::-webkit-details-marker { display: none; }
.steps-icon { color: #7c5cfc; }
.ctx-icon { color: #2196f3; }

.steps-list { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
.step-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
}
.step-idx {
  width: 18px; height: 18px;
  border-radius: 50%;
  background: #1e1e30;
  color: #888;
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; font-weight: 700; flex-shrink: 0;
}
.step-module { color: #e0e0e0; font-family: monospace; }
.step-title { color: #777; font-style: italic; }

.ctx-list { margin-top: 8px; display: flex; flex-direction: column; gap: 10px; }
.ctx-source { display: flex; flex-direction: column; gap: 4px; }
.ctx-source-name {
  font-size: 11px; font-weight: 700;
  color: #2196f3; text-transform: uppercase; letter-spacing: 0.04em;
}
.ctx-chunk {
  font-size: 12px; color: #999; line-height: 1.5;
  padding: 6px 8px;
  background: #13131f;
  border-radius: 4px;
}
.ctx-more { font-size: 11px; color: #555; }

/* Streaming */
.streaming-indicator { display: flex; gap: 4px; padding: 4px 0; }
.dot {
  width: 6px; height: 6px;
  background: #7c5cfc;
  border-radius: 50%;
  animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
  0%, 80%, 100% { transform: scale(0.7); opacity: 0.5; }
  40% { transform: scale(1); opacity: 1; }
}

.bubble-error {
  font-size: 13px;
  color: #ff9999;
  background: rgba(255,85,85,0.08);
  border: 1px solid rgba(255,85,85,0.2);
  border-radius: 6px;
  padding: 8px 12px;
}

/* Feedback */
.feedback-block {
  background: #0f1a2e;
  border: 1px solid #1a3a60;
  border-radius: 8px;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.feedback-header {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #e0e0e0;
}
.feedback-icon { font-size: 16px; }
.feedback-module {
  background: #1a3a60;
  color: #6ab0ff;
  font-size: 11px;
  padding: 2px 7px;
  border-radius: 99px;
  font-family: monospace;
}
.feedback-question {
  font-size: 14px;
  color: #c0c0d0;
  margin: 0;
  line-height: 1.5;
}

/* Steps / trace panel */
.steps-panel {
  width: 0;
  overflow: hidden;
  background: #0f0f1c;
  border-left: 1px solid #1e1e30;
  transition: width 0.25s ease;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}
.steps-panel--open { width: 280px; }
.steps-panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 14px;
  border-bottom: 1px solid #1e1e30;
  flex-shrink: 0;
}
.steps-panel-title { font-size: 13px; font-weight: 700; color: #888; }
.steps-panel-close {
  background: none; border: none; color: #555; cursor: pointer; font-size: 14px;
}
.steps-panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.steps-empty { font-size: 12px; color: #444; text-align: center; padding: 20px 0; }
.trace-entry {
  background: #13131f;
  border: 1px solid #1e1e30;
  border-radius: 6px;
  padding: 8px 10px;
}
.trace-entry-header {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
}
.trace-q-label {
  color: #555;
  background: #1e1e30;
  padding: 1px 5px;
  border-radius: 4px;
}
.trace-step-idx { color: #666; }
.trace-module { color: #e0e0e0; font-family: monospace; font-size: 12px; flex: 1; }
.trace-title { font-size: 11px; color: #666; font-style: italic; margin-top: 2px; }
.trace-section-title {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #555;
  margin-top: 8px;
}
.trace-ctx-entry {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  padding: 6px 10px;
  background: #13131f;
  border: 1px solid #1e1e30;
  border-radius: 6px;
}
.trace-ctx-source { color: #2196f3; font-family: monospace; }
.trace-ctx-count { color: #555; font-size: 11px; }

/* ── Input bar ── */
.input-bar {
  display: flex;
  align-items: flex-end;
  gap: 10px;
  padding: 12px 16px;
  background: #0f0f1c;
  border-top: 1px solid #1e1e30;
  flex-shrink: 0;
}
.trace-toggle {
  background: transparent;
  border: 1px solid #333;
  color: #777;
  padding: 7px 12px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
  flex-shrink: 0;
  transition: all 0.15s;
}
.trace-toggle:hover { background: #1a1a2e; color: #ccc; }
.trace-toggle--active { background: #1a1a2e; border-color: #7c5cfc; color: #7c5cfc; }

.input-wrap { flex: 1; }
.question-input {
  width: 100%;
  background: #1a1a2e;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 10px 14px;
  color: #e0e0e0;
  font-size: 14px;
  font-family: inherit;
  resize: none;
  outline: none;
  line-height: 1.5;
  max-height: 200px;
  overflow-y: auto;
  transition: border-color 0.15s;
}
.question-input:focus { border-color: #2196f3; }
.question-input:disabled { opacity: 0.5; cursor: not-allowed; }

.send-btn {
  background: #2196f3;
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  flex-shrink: 0;
  min-width: 80px;
  transition: background 0.15s;
  display: flex;
  align-items: center;
  justify-content: center;
}
.send-btn:hover:not(:disabled) { background: #1976d2; }
.send-btn:disabled { opacity: 0.4; cursor: not-allowed; }
.sending-dots { display: flex; gap: 3px; align-items: center; }

/* Transitions */
.msg-enter-active { transition: all 0.2s ease; }
.msg-enter-from { opacity: 0; transform: translateY(8px); }
</style>
