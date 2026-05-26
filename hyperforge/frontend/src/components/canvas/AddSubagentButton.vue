<template>
  <div class="add-subagent" @click.stop>
    <button class="add-btn" :title="title" @click="open = !open">+</button>

    <div v-if="open" class="popover" @click.stop>
      <!-- Step 1: pick a link type if more than one is allowed -->
      <div v-if="allowedLinkTypes.length > 1 && !chosenLinkType" class="link-type-row">
        <div class="popover-label">Connect as…</div>
        <button
          v-for="lt in allowedLinkTypes"
          :key="lt"
          class="link-type-btn"
          @click="chosenLinkType = lt"
        >
          {{ linkTypeLabel(lt) }}
        </button>
      </div>

      <!-- Step 2: agent palette -->
      <template v-else>
        <div class="popover-header">
          <span class="popover-label">{{ linkTypeLabel(effectiveLinkType) }}</span>
          <button v-if="allowedLinkTypes.length > 1" class="link-back" @click="chosenLinkType = null">
            change
          </button>
        </div>

        <input
          ref="searchEl"
          v-model="search"
          class="palette-search"
          placeholder="Search…"
          @keydown.escape="close"
        />
        <div class="palette-list">
          <div
            v-for="schema in filteredSchemas"
            :key="schema.id"
            class="palette-item"
            @click="pickAgent(schema.id)"
          >
            <span class="palette-item-id">{{ schema.id }}</span>
            <span class="palette-item-desc">{{ schema.title }}</span>
          </div>
          <div v-if="filteredSchemas.length === 0" class="palette-empty">No results</div>
        </div>
        <button class="palette-close" @click="close">Cancel</button>
      </template>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import { useSchemaStore } from '@/stores/schema'
import { useWorkflowStore } from '@/stores/workflow'
import type { ConnectableKey, StageType } from '@/types/arag'

const props = defineProps<{
  parentNodeId: string
  /** Stage of the parent node (used to filter the palette by default). */
  parentStage: StageType
  /** Connectable keys allowed for this parent (per its schema). */
  allowedLinkTypes: ConnectableKey[]
}>()

const schemaStore = useSchemaStore()
const workflowStore = useWorkflowStore()

const open = ref(false)
const search = ref('')
const chosenLinkType = ref<ConnectableKey | null>(null)
const searchEl = ref<HTMLInputElement | null>(null)

const title = computed(() => 'Add subagent')

/** When only one link type is allowed, skip the picker and use it directly. */
const effectiveLinkType = computed<ConnectableKey | null>(() => {
  if (props.allowedLinkTypes.length === 1) return props.allowedLinkTypes[0]
  return chosenLinkType.value
})

watch(
  () => effectiveLinkType.value,
  async (lt) => {
    if (lt && open.value) {
      await nextTick()
      searchEl.value?.focus()
    }
  },
)

watch(open, (isOpen) => {
  if (!isOpen) {
    chosenLinkType.value = null
    search.value = ''
  }
})

const LINK_TYPE_LABELS: Record<ConnectableKey, string> = {
  then: 'Then',
  else_: 'Else',
  fallback: 'Fallback',
  next_agent: 'Next agent',
  agents: 'Add agent',
  registered_agents: 'Register agent',
}

function linkTypeLabel(lt: ConnectableKey | null): string {
  return lt ? LINK_TYPE_LABELS[lt] : ''
}

/** Stage to filter the palette by, given the chosen link type. */
const paletteStage = computed<StageType | null>(() => {
  const lt = effectiveLinkType.value
  if (!lt) return null
  if (lt === 'registered_agents' || lt === 'agents') return 'context'
  return props.parentStage
})

const filteredSchemas = computed(() => {
  const stage = paletteStage.value
  if (!stage) return []
  const q = search.value.toLowerCase()
  return schemaStore
    .allAgentsForStage(stage)
    .filter(
      (schema) =>
        !q ||
        schema.id.toLowerCase().includes(q) ||
        schema.title.toLowerCase().includes(q),
    )
})

/** 6-char alphanumeric suffix for stable child ids on registered_agents. */
function shortId(): string {
  return Math.random().toString(36).slice(2, 8)
}

function pickAgent(moduleId: string) {
  const lt = effectiveLinkType.value
  if (!lt) return
  const newPath = workflowStore.addChild(props.parentNodeId, lt, moduleId)
  if (!newPath) return
  // Smart-agent children must carry a stable `id` because the parent's
  // registered_agents_descriptions / ..._exposed_functions dicts are keyed
  // by `child.context_config.id` (see smart/agent.py).
  if (lt === 'registered_agents') {
    workflowStore.updateNodeConfig(newPath, { id: `${moduleId}_${shortId()}` })
  }
  workflowStore.selectNode(newPath)
  close()
}

function close() {
  open.value = false
}
</script>

<style scoped>
.add-subagent {
  position: relative;
}

.add-btn {
  background: #2a2a40;
  border: 1px solid #444;
  color: #7c5cfc;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  font-size: 14px;
  font-weight: 700;
  line-height: 1;
  cursor: pointer;
  padding: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  transition: background 0.15s, border-color 0.15s;
}

.add-btn:hover {
  background: #3a3a55;
  border-color: #7c5cfc;
}

.popover {
  position: absolute;
  top: 28px;
  right: 0;
  z-index: 50;
  width: 240px;
  max-height: 320px;
  background: #1a1a2e;
  border: 1px solid #2a2a40;
  border-radius: 8px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}

.popover-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.popover-label {
  font-size: 11px;
  font-weight: 700;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.link-type-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.link-type-btn {
  background: #13131f;
  border: 1px solid #333;
  color: #ddd;
  padding: 6px 10px;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  text-align: left;
}

.link-type-btn:hover {
  border-color: #7c5cfc;
  background: #1e1e30;
}

.link-back {
  background: none;
  border: none;
  color: #666;
  font-size: 11px;
  cursor: pointer;
  text-decoration: underline;
}

.link-back:hover {
  color: #aaa;
}

.palette-search {
  background: #13131f;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 6px 10px;
  color: #e0e0e0;
  font-size: 12px;
  outline: none;
}

.palette-search:focus {
  border-color: #7c5cfc;
}

.palette-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 200px;
}

.palette-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  transition: background 0.1s;
}

.palette-item:hover {
  background: #252540;
}

.palette-item-id {
  font-size: 12px;
  color: #e0e0e0;
  font-family: monospace;
  flex-shrink: 0;
}

.palette-item-desc {
  font-size: 11px;
  color: #666;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.palette-empty {
  font-size: 12px;
  color: #444;
  text-align: center;
  padding: 8px;
}

.palette-close {
  background: transparent;
  border: 1px solid #333;
  color: #888;
  padding: 5px 10px;
  border-radius: 5px;
  font-size: 12px;
  cursor: pointer;
  align-self: flex-end;
}
</style>
