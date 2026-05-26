<template>
  <div v-if="nodeId && nodeConfig" class="config-panel">
    <div class="panel-header">
      <span class="stage-badge" :style="{ background: STAGE_COLORS[stage] }">
        {{ STAGE_LABELS[stage] }}
      </span>
      <h3 class="panel-title">{{ nodeConfig.module }}</h3>
      <button class="close-btn" @click="store.selectNode(null)">✕</button>
    </div>

    <div class="panel-body">
      <!-- Node title override -->
      <div class="field-group">
        <label class="field-label">Title (display name)</label>
        <input
          class="field-input"
          :value="(nodeConfig.title as string) || ''"
          placeholder="Optional display name"
          @input="patch('title', ($event.target as HTMLInputElement).value)"
        />
      </div>

      <!-- Smart-agent extras: per-child planner description / exposed functions -->
      <div v-if="registeredAgentExtras" class="field-group">
        <label class="field-label">
          Registered agent id
        </label>
        <div class="field-readonly">{{ registeredAgentExtras.childKey }}</div>
        <label class="field-label" style="margin-top: 8px">
          Description (shown to the planner)
        </label>
        <textarea
          class="field-input field-textarea"
          rows="2"
          :value="registeredAgentExtras.description"
          placeholder="What this agent is good at"
          @input="patchRegisteredAgentDescription(($event.target as HTMLTextAreaElement).value)"
        />
        <label class="field-label" style="margin-top: 8px">
          Exposed functions (one per line)
        </label>
        <textarea
          class="field-input field-textarea"
          rows="3"
          :value="registeredAgentExtras.exposedFunctionsText"
          placeholder="search&#10;summarize"
          @input="patchRegisteredAgentExposedFunctions(($event.target as HTMLTextAreaElement).value)"
        />
      </div>

      <div v-if="agentSchema" class="schema-section">
        <h4 class="section-title">Configuration</h4>
        <SchemaForm
          :schema="agentSchema.config_schema"
          :value="nodeConfig"
          :stage="stage"
          :defs="schemaStore.defs"
          @update="onFormUpdate"
        />
      </div>
      <div v-else class="no-schema">
        <p>No schema found for module <code>{{ nodeConfig.module }}</code>.</p>
        <p class="hint">Raw config:</p>
        <pre class="raw-json">{{ JSON.stringify(nodeConfig, null, 2) }}</pre>
      </div>
    </div>

    <div class="panel-footer">
      <button class="btn-danger" @click="onRemove">Remove node</button>
    </div>
  </div>

  <div v-else class="panel-empty">
    <div class="empty-icon">⬡</div>
    <p>Click a node to configure it</p>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useWorkflowStore } from '@/stores/workflow'
import { useSchemaStore } from '@/stores/schema'
import { STAGE_COLORS, STAGE_LABELS } from '@/types/arag'
import type { AgentStepConfig, StageType } from '@/types/arag'
import SchemaForm from './SchemaForm.vue'

const store = useWorkflowStore()
const schemaStore = useSchemaStore()

const nodeId = computed(() => store.selectedNodeId)
const nodeConfig = computed(() => (nodeId.value ? store.getNodeConfig(nodeId.value) : null))

/** The canvas-node descriptor (carries parentId + parentLinkType). */
const canvasNode = computed(() => {
  if (!nodeId.value) return null
  return store.canvasNodes.find((n) => n.id === nodeId.value) ?? null
})

const stage = computed<StageType>(() => canvasNode.value?.stage ?? 'context')

const agentSchema = computed(() => {
  if (!nodeConfig.value) return null
  return schemaStore.getAgentSchema(stage.value, nodeConfig.value.module as string)
})

// ── Smart-agent extras ─────────────────────────────────────────────────────
//
// `SmartAgentConfig` keeps two parent-side maps keyed by the child node id:
//   - registered_agents_descriptions: { <id>: <text> }
//   - registered_agents_exposed_functions: { <id>: list[str] }
// These are NOT shown on the child schema — they're properties of the parent
// indexed by the child. We surface them here when the selected node was
// reached through `registered_agents`.

interface RegisteredAgentExtras {
  parentNodeId: string
  childKey: string
  description: string
  exposedFunctionsText: string
}

const registeredAgentExtras = computed<RegisteredAgentExtras | null>(() => {
  const node = canvasNode.value
  if (!node || node.parentLinkType !== 'registered_agents' || !node.parentId) return null

  const parent = store.getNodeConfig(node.parentId) as AgentStepConfig | null
  if (!parent) return null

  // Smart-agent parent maps are keyed strictly by the child's `id`. If the
  // child has no id (shouldn't happen — `AddSubagentButton` always assigns
  // one), the maps cannot be addressed and we don't surface the panel.
  const childKey = (nodeConfig.value as AgentStepConfig & { id?: string })?.id
  if (!childKey) return null

  const descMap =
    (parent['registered_agents_descriptions'] as Record<string, string> | undefined) ?? {}
  const fnMap =
    (parent['registered_agents_exposed_functions'] as Record<string, string[]> | undefined) ?? {}

  return {
    parentNodeId: node.parentId,
    childKey,
    description: descMap[childKey] ?? '',
    exposedFunctionsText: (fnMap[childKey] ?? []).join('\n'),
  }
})

function patchRegisteredAgentDescription(text: string) {
  const extras = registeredAgentExtras.value
  if (!extras) return
  const parent = store.getNodeConfig(extras.parentNodeId) as AgentStepConfig | null
  if (!parent) return
  const descMap = {
    ...((parent['registered_agents_descriptions'] as Record<string, string> | undefined) ?? {}),
    [extras.childKey]: text,
  }
  store.updateNodeConfig(extras.parentNodeId, {
    registered_agents_descriptions: descMap,
  })
}

function patchRegisteredAgentExposedFunctions(text: string) {
  const extras = registeredAgentExtras.value
  if (!extras) return
  const parent = store.getNodeConfig(extras.parentNodeId) as AgentStepConfig | null
  if (!parent) return
  const fns = text.split('\n').map((s) => s.trim()).filter(Boolean)
  const fnMap = {
    ...((parent['registered_agents_exposed_functions'] as Record<string, string[]> | undefined) ??
      {}),
    [extras.childKey]: fns,
  }
  store.updateNodeConfig(extras.parentNodeId, {
    registered_agents_exposed_functions: fnMap,
  })
}

function patch(key: string, value: unknown) {
  if (!nodeId.value) return
  store.updateNodeConfig(nodeId.value, { [key]: value })
}

function onFormUpdate(updates: Record<string, unknown>) {
  if (!nodeId.value) return
  store.updateNodeConfig(nodeId.value, updates)
}

function onRemove() {
  if (!nodeId.value) return
  store.removeStep(nodeId.value)
}
</script>

<style scoped>
.config-panel {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  background: #1a1a2e;
}

.panel-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 14px 16px;
  border-bottom: 1px solid #2a2a40;
  flex-shrink: 0;
}

.stage-badge {
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  padding: 2px 8px;
  border-radius: 99px;
  color: #fff;
  flex-shrink: 0;
}

.panel-title {
  flex: 1;
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: #e0e0e0;
  font-family: monospace;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.close-btn {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 14px;
  flex-shrink: 0;
  transition: color 0.1s;
}

.close-btn:hover {
  color: #ccc;
}

.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.field-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.field-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  color: #888;
}

.field-input {
  background: #13131f;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 7px 10px;
  color: #e0e0e0;
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
}

.field-input:focus {
  border-color: #7c5cfc;
}

.field-textarea {
  resize: vertical;
  font-family: inherit;
  line-height: 1.4;
  min-height: 40px;
}

.field-readonly {
  background: #13131f;
  border: 1px solid #2a2a3e;
  border-radius: 6px;
  padding: 6px 10px;
  color: #888;
  font-family: monospace;
  font-size: 12px;
  user-select: text;
}

.section-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #666;
  margin: 0 0 10px;
}

.schema-section {
  display: flex;
  flex-direction: column;
}

.no-schema {
  color: #888;
  font-size: 13px;
}

.no-schema code {
  background: #252535;
  padding: 1px 5px;
  border-radius: 4px;
  font-family: monospace;
}

.hint {
  margin-top: 10px;
  font-size: 11px;
}

.raw-json {
  font-size: 11px;
  font-family: monospace;
  background: #13131f;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 10px;
  overflow-x: auto;
  color: #aaa;
  white-space: pre-wrap;
  word-break: break-all;
}

.panel-footer {
  padding: 12px 16px;
  border-top: 1px solid #2a2a40;
  flex-shrink: 0;
}

.btn-danger {
  width: 100%;
  background: transparent;
  border: 1px solid #ff5555;
  color: #ff5555;
  padding: 7px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: background 0.15s;
}

.btn-danger:hover {
  background: rgba(255, 85, 85, 0.1);
}

.panel-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #555;
  gap: 10px;
}

.empty-icon {
  font-size: 48px;
  opacity: 0.3;
}

.panel-empty p {
  font-size: 13px;
  text-align: center;
}
</style>
