<template>
  <div class="app-shell">
    <nav class="app-nav">
      <RouterLink to="/" class="nav-link" active-class="nav-link--active">
        <span class="nav-icon">⬡</span> Editor
      </RouterLink>
      <RouterLink to="/chat" class="nav-link" active-class="nav-link--active">
        <span class="nav-icon">💬</span> Chat
      </RouterLink>
      <RouterLink to="/sources" class="nav-link" active-class="nav-link--active">
        <span class="nav-icon">⛁</span> Sources
      </RouterLink>
    </nav>
    <div class="app-content">
      <RouterView />
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted } from 'vue'
import { useWorkflowStore } from '@/stores/workflow'
import { useSchemaStore } from '@/stores/schema'

const workflowStore = useWorkflowStore()
const schemaStore = useSchemaStore()

onMounted(async () => {
  // /api/v1/ui/schema returns a single payload that includes the merged $defs,
  // so a second round-trip is no longer needed.
  await Promise.all([workflowStore.load(), schemaStore.load()])
})
</script>

<style>
/* ── Global reset ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, #app {
  margin: 0; padding: 0; height: 100%;
  width: 100%; overflow: hidden;
  background: #13131f; color: #e0e0e0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
}
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }
</style>

<style scoped>
.app-shell {
  display: flex;
  flex-direction: row;
  height: 100vh;
  overflow: hidden;
}

.app-nav {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 52px;
  background: #0a0a14;
  border-right: 1px solid #1e1e30;
  padding: 12px 0;
  gap: 4px;
  flex-shrink: 0;
  z-index: 100;
}

.nav-link {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 40px;
  height: 40px;
  border-radius: 8px;
  text-decoration: none;
  color: #666;
  font-size: 9px;
  gap: 2px;
  transition: background 0.15s, color 0.15s;
}

.nav-link:hover { background: #1a1a2e; color: #bbb; }
.nav-link--active { background: #1a1a2e; color: #7c5cfc; }

.nav-icon { font-size: 16px; line-height: 1; }

.app-content {
  flex: 1;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}
</style>
