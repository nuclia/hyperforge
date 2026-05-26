<template>
  <div
    class="agent-node"
    :class="[`stage-${data.stage}`, { selected }]"
  >
    <Handle type="target" :position="Position.Left" />
    <div class="node-header">
      <span class="stage-badge" :style="{ background: stageColor }">
        {{ stageLabel }}
      </span>
      <div class="node-actions">
        <AddSubagentButton
          v-if="allowedLinkTypes.length > 0"
          :parent-node-id="id"
          :parent-stage="data.stage"
          :allowed-link-types="allowedLinkTypes"
        />
        <button class="delete-btn" title="Remove" @click.stop="onDelete">✕</button>
      </div>
    </div>
    <div class="node-body">
      <div class="node-title">{{ nodeTitle }}</div>
      <div class="node-module">{{ data.config.module }}</div>
    </div>
    <Handle type="source" :position="Position.Right" />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Handle, Position } from '@vue-flow/core'
import { useWorkflowStore } from '@/stores/workflow'
import { useSchemaStore } from '@/stores/schema'
import { CONNECTABLE_KEYS, STAGE_COLORS, STAGE_LABELS } from '@/types/arag'
import type { AgentStepConfig, ConnectableKey, StageType } from '@/types/arag'
import AddSubagentButton from '@/components/canvas/AddSubagentButton.vue'

const props = defineProps<{
  id: string
  data: {
    stage: StageType
    stepIndex: number
    config: AgentStepConfig
  }
  selected?: boolean
}>()

const store = useWorkflowStore()
const schemaStore = useSchemaStore()

const stageColor = computed(() => STAGE_COLORS[props.data.stage])
const stageLabel = computed(() => STAGE_LABELS[props.data.stage])
const nodeTitle = computed(
  () =>
    (props.data.config.title as string | undefined) ||
    (props.data.config.module as string),
)

/**
 * Connectable keys advertised by this module's config schema.
 * The "+" button is only shown when at least one is present.
 */
const allowedLinkTypes = computed<ConnectableKey[]>(() => {
  const def = schemaStore.defForModule(props.data.config.module as string)
  if (!def?.properties) return []
  return CONNECTABLE_KEYS.filter((k) => k in def.properties!) as ConnectableKey[]
})

function onDelete() {
  store.removeStep(props.id)
}
</script>

<style scoped>
.agent-node {
  background: #1e1e30;
  border: 2px solid #333;
  border-radius: 10px;
  min-width: 180px;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s;
  user-select: none;
}

.agent-node.selected {
  border-color: #7c5cfc;
  box-shadow: 0 0 0 3px rgba(124, 92, 252, 0.3);
}

.agent-node:hover {
  border-color: #555;
}

.node-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 8px 0;
}

.node-actions {
  display: flex;
  align-items: center;
  gap: 6px;
}

.stage-badge {
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #fff;
  padding: 2px 7px;
  border-radius: 99px;
}

.delete-btn {
  background: none;
  border: none;
  color: #666;
  cursor: pointer;
  font-size: 12px;
  padding: 0 2px;
  line-height: 1;
  transition: color 0.1s;
}

.delete-btn:hover {
  color: #ff5555;
}

.node-body {
  padding: 8px 12px 12px;
}

.node-title {
  font-size: 14px;
  font-weight: 600;
  color: #e0e0e0;
  margin-bottom: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 160px;
}

.node-module {
  font-size: 11px;
  color: #888;
  font-family: monospace;
}
</style>
