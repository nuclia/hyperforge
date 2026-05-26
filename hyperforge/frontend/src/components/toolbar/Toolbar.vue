<template>
  <div class="toolbar">
    <!-- Left: agent selector -->
    <div class="toolbar-left">
      <span class="toolbar-brand">ARAG Studio</span>
      <select
        v-if="workflowStore.agentIds.length > 0"
        class="agent-select"
        :value="workflowStore.activeAgentId ?? ''"
        @change="workflowStore.setActiveAgent(($event.target as HTMLSelectElement).value)"
      >
        <option v-for="id in workflowStore.agentIds" :key="id" :value="id">
          {{ agentTitle(id) }}
        </option>
      </select>

      <select
        v-if="workflowIds.length > 0"
        class="workflow-select"
        :value="workflowStore.activeWorkflowId"
        @change="workflowStore.setActiveWorkflow(($event.target as HTMLSelectElement).value)"
      >
        <option v-for="id in workflowIds" :key="id" :value="id">
          {{ workflowLabel(id) }}
        </option>
      </select>
    </div>

    <!-- Right: actions -->
    <div class="toolbar-right">
      <button class="btn btn-secondary" @click="openAddAgent">+ Add Agent</button>

      <button
        class="btn btn-primary"
        :class="{ saving: workflowStore.saving }"
        :disabled="workflowStore.saving || !workflowStore.dirty"
        @click="onSave"
      >
        {{ workflowStore.saving ? 'Saving…' : workflowStore.dirty ? 'Save *' : 'Saved' }}
      </button>

      <button class="btn btn-secondary" title="Export config JSON" @click="onExport">
        Export
      </button>
    </div>

    <!-- Agent palette modal -->
    <Teleport to="body">
      <div v-if="showPalette" class="modal-backdrop" @click.self="showPalette = false">
        <div class="modal">
          <div class="modal-header">
            <h3 class="modal-title">Add Agent</h3>
            <button class="close-btn" @click="showPalette = false">✕</button>
          </div>

          <div class="modal-body">
            <div class="palette-search">
              <input
                v-model="search"
                class="search-input"
                placeholder="Search agents…"
                autofocus
              />
            </div>

            <div v-for="stage in STAGE_ORDER" :key="stage" class="palette-stage">
              <div v-if="filteredForStage(stage).length > 0">
                <h4
                  class="palette-stage-label"
                  :style="{ color: STAGE_COLORS[stage] }"
                >
                  {{ STAGE_LABELS[stage] }}
                </h4>
                <div class="palette-grid">
                  <button
                    v-for="schema in filteredForStage(stage)"
                    :key="schema.id"
                    class="palette-card"
                    @click="addAgent(stage, schema.id)"
                  >
                    <div class="palette-card-title">{{ schema.title }}</div>
                    <div class="palette-card-desc">{{ schema.description }}</div>
                  </button>
                </div>
              </div>
            </div>

            <div v-if="allFiltered.length === 0" class="palette-empty">
              No agents found for "{{ search }}"
            </div>
          </div>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useWorkflowStore } from '@/stores/workflow'
import { useSchemaStore } from '@/stores/schema'
import { STAGE_ORDER, STAGE_COLORS, STAGE_LABELS } from '@/types/arag'
import type { StageType, AgentSchema } from '@/types/arag'

const workflowStore = useWorkflowStore()
const schemaStore = useSchemaStore()

const showPalette = ref(false)
const search = ref('')

const workflowIds = computed(() => {
  if (!workflowStore.activeAgent) return []
  return Object.keys(workflowStore.activeAgent.workflows)
})

function agentTitle(agentId: string): string {
  return workflowStore.config[agentId]?.title || agentId
}

function workflowLabel(id: string): string {
  const wf = workflowStore.activeAgent?.workflows[id]
  return wf?.name || id
}

function openAddAgent() {
  search.value = ''
  showPalette.value = true
}

function filteredForStage(stage: StageType): AgentSchema[] {
  const all = schemaStore.allAgentsForStage(stage)
  if (!search.value) return all
  const q = search.value.toLowerCase()
  return all.filter(
    (s: AgentSchema) =>
      s.id.toLowerCase().includes(q) ||
      s.title.toLowerCase().includes(q) ||
      s.description.toLowerCase().includes(q),
  )
}

const allFiltered = computed(() =>
  STAGE_ORDER.flatMap((s: StageType) => filteredForStage(s)),
)

function addAgent(stage: StageType, moduleId: string) {
  workflowStore.addStep(stage, { module: moduleId })
  showPalette.value = false
}

async function onSave() {
  try {
    await workflowStore.save()
  } catch {
    // error is stored in workflowStore.error
  }
}

function onExport() {
  const data = JSON.stringify(workflowStore.config, null, 2)
  const blob = new Blob([data], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'arag-config.json'
  a.click()
  URL.revokeObjectURL(url)
}
</script>

<style scoped>
.toolbar {
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

.toolbar-left,
.toolbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}

.toolbar-brand {
  font-size: 15px;
  font-weight: 800;
  color: #7c5cfc;
  letter-spacing: 0.02em;
  white-space: nowrap;
}

.agent-select,
.workflow-select {
  background: #1a1a2e;
  border: 1px solid #333;
  color: #ddd;
  border-radius: 6px;
  padding: 5px 10px;
  font-size: 13px;
  cursor: pointer;
  outline: none;
  max-width: 180px;
}

.btn {
  padding: 6px 14px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all 0.15s;
}

.btn-primary {
  background: #7c5cfc;
  color: #fff;
}

.btn-primary:hover:not(:disabled) {
  background: #6a4de0;
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: default;
}

.btn-secondary {
  background: transparent;
  color: #bbb;
  border: 1px solid #333;
}

.btn-secondary:hover {
  background: #1a1a2e;
  color: #fff;
}

/* ── Modal ── */

.modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.7);
  z-index: 1000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.modal {
  background: #1a1a2e;
  border: 1px solid #2a2a40;
  border-radius: 12px;
  width: 700px;
  max-width: 100%;
  max-height: 80vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.modal-header {
  display: flex;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #2a2a40;
  flex-shrink: 0;
}

.modal-title {
  flex: 1;
  margin: 0;
  font-size: 16px;
  font-weight: 700;
  color: #e0e0e0;
}

.close-btn {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 16px;
  transition: color 0.1s;
}

.close-btn:hover {
  color: #ccc;
}

.modal-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.palette-search {
  flex-shrink: 0;
}

.search-input {
  width: 100%;
  box-sizing: border-box;
  background: #13131f;
  border: 1px solid #333;
  border-radius: 8px;
  padding: 8px 12px;
  color: #e0e0e0;
  font-size: 14px;
  outline: none;
  transition: border-color 0.15s;
}

.search-input:focus {
  border-color: #7c5cfc;
}

.palette-stage-label {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin: 0 0 10px;
}

.palette-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 8px;
}

.palette-card {
  background: #13131f;
  border: 1px solid #2a2a40;
  border-radius: 8px;
  padding: 10px 12px;
  text-align: left;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
}

.palette-card:hover {
  border-color: #7c5cfc;
  background: #1a1a2e;
}

.palette-card-title {
  font-size: 13px;
  font-weight: 600;
  color: #e0e0e0;
  margin-bottom: 4px;
}

.palette-card-desc {
  font-size: 11px;
  color: #777;
  line-height: 1.4;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.palette-empty {
  font-size: 13px;
  color: #555;
  text-align: center;
  padding: 20px;
}
</style>
