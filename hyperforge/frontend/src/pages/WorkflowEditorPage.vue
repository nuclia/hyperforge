<template>
  <div class="editor-page">
    <Toolbar />

    <div v-if="workflowStore.loading" class="loading-screen">
      <div class="spinner" />
      <p>Loading config…</p>
    </div>

    <div v-else-if="workflowStore.error" class="error-screen">
      <p class="error-msg">{{ workflowStore.error }}</p>
      <button class="btn-retry" @click="workflowStore.load()">Retry</button>
    </div>

    <div v-else class="workspace">
      <div class="canvas-area">
        <WorkflowCanvas v-if="workflowStore.activeAgent" />
        <div v-else class="no-agent-hint">
          <div class="hint-icon">⬡</div>
          <p>No agent loaded. Set <code>AGENTS_CONFIG</code> and restart.</p>
        </div>
      </div>
      <div class="sidebar-right">
        <ConfigPanel />
      </div>
    </div>

    <Transition name="fade">
      <div v-if="workflowStore.error && !workflowStore.loading" class="toast error-toast">
        {{ workflowStore.error }}
      </div>
    </Transition>
  </div>
</template>

<script setup lang="ts">
import { useWorkflowStore } from '@/stores/workflow'
import Toolbar from '@/components/toolbar/Toolbar.vue'
import WorkflowCanvas from '@/components/canvas/WorkflowCanvas.vue'
import ConfigPanel from '@/components/sidebar/ConfigPanel.vue'

const workflowStore = useWorkflowStore()
</script>

<style scoped>
.editor-page {
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.loading-screen, .error-screen {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 16px;
  color: #888;
}

.spinner {
  width: 40px; height: 40px;
  border: 3px solid #333;
  border-top-color: #7c5cfc;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.error-msg { color: #ff5555; font-size: 14px; }
.btn-retry {
  background: #7c5cfc; color: #fff; border: none;
  padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 14px;
}

.workspace {
  flex: 1;
  display: grid;
  grid-template-columns: 1fr 320px;
  overflow: hidden;
}

.canvas-area {
  position: relative;
  overflow: hidden;
  border-right: 1px solid #1e1e30;
}

.sidebar-right { overflow: hidden; }

.no-agent-hint {
  display: flex; flex-direction: column;
  align-items: center; justify-content: center;
  height: 100%; color: #555; gap: 12px;
}
.hint-icon { font-size: 64px; opacity: 0.2; }
.no-agent-hint p { font-size: 14px; text-align: center; }
.no-agent-hint code {
  background: #1e1e30; padding: 2px 6px;
  border-radius: 4px; font-family: monospace;
}

.toast {
  position: fixed; bottom: 20px; left: 50%;
  transform: translateX(-50%);
  padding: 10px 20px; border-radius: 8px;
  font-size: 13px; z-index: 9999; max-width: 400px;
}
.error-toast {
  background: #3a1a1a; border: 1px solid #ff5555; color: #ff9999;
}
.fade-enter-active, .fade-leave-active { transition: opacity 0.3s; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
</style>
