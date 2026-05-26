<template>
  <div class="driver-select">
    <template v-if="multiple">
      <!-- Multi-select: render chips + add dropdown -->
      <div class="multi-wrap">
        <div v-if="selected.length > 0" class="chips">
          <span v-for="id in selected" :key="id" class="chip">
            {{ labelFor(id) }}
            <button class="chip-remove" type="button" @click="deselect(id)">×</button>
          </span>
        </div>
        <select
          class="driver-select-input"
          value=""
          @change="selectOne(($event.target as HTMLSelectElement).value); ($event.target as HTMLSelectElement).value = ''"
        >
          <option value="" disabled>Add driver…</option>
          <option
            v-for="d in availableDrivers"
            :key="d.identifier"
            :value="d.identifier"
            :disabled="selected.includes(d.identifier)"
          >
            {{ d.name || d.identifier }}
            <template v-if="d.name !== d.identifier"> ({{ d.identifier }})</template>
          </option>
        </select>
      </div>
    </template>

    <template v-else>
      <!-- Single select -->
      <select
        class="driver-select-input"
        :value="modelValue as string"
        @change="emit('update:modelValue', ($event.target as HTMLSelectElement).value)"
      >
        <option value="">None</option>
        <option
          v-for="d in availableDrivers"
          :key="d.identifier"
          :value="d.identifier"
        >
          {{ d.name || d.identifier }}
          <template v-if="d.name !== d.identifier"> ({{ d.identifier }})</template>
        </option>
      </select>
    </template>

    <!-- Warning when no drivers exist -->
    <div v-if="availableDrivers.length === 0" class="no-drivers-warn">
      No drivers configured.
      <RouterLink to="/sources" class="warn-link">Add one in Sources →</RouterLink>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink } from 'vue-router'
import { useWorkflowStore } from '@/stores/workflow'

const props = defineProps<{
  modelValue: string | string[]
  /** If true, allows selecting multiple drivers (value = string[]) */
  multiple?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', val: string | string[]): void
}>()

const workflowStore = useWorkflowStore()

const availableDrivers = computed(() => workflowStore.allDrivers)

const selected = computed((): string[] => {
  if (!props.multiple) return []
  if (Array.isArray(props.modelValue)) return props.modelValue as string[]
  return props.modelValue ? [props.modelValue as string] : []
})

function labelFor(identifier: string): string {
  const d = availableDrivers.value.find((x) => x.identifier === identifier)
  return d?.name || identifier
}

function selectOne(id: string) {
  if (!id) return
  const arr = [...selected.value]
  if (!arr.includes(id)) arr.push(id)
  emit('update:modelValue', arr)
}

function deselect(id: string) {
  emit('update:modelValue', selected.value.filter((x) => x !== id))
}
</script>

<style scoped>
.driver-select { display: flex; flex-direction: column; gap: 6px; }

.driver-select-input {
  background: #13131f; border: 1px solid #333; border-radius: 6px;
  padding: 7px 10px; color: #e0e0e0; font-size: 13px; outline: none;
  transition: border-color 0.15s; width: 100%; box-sizing: border-box; cursor: pointer;
}
.driver-select-input:focus { border-color: #7c5cfc; }

.multi-wrap { display: flex; flex-direction: column; gap: 6px; }

.chips { display: flex; flex-wrap: wrap; gap: 5px; }

.chip {
  display: inline-flex; align-items: center; gap: 4px;
  background: #1e2a3a; color: #7eb8f7;
  padding: 3px 8px; border-radius: 99px; font-size: 12px;
}

.chip-remove {
  background: none; border: none; cursor: pointer;
  color: #5b8bb5; font-size: 14px; line-height: 1; padding: 0;
  transition: color 0.1s;
}
.chip-remove:hover { color: #ff5555; }

.no-drivers-warn {
  font-size: 11px;
  color: #ff9800;
  display: flex;
  align-items: center;
  gap: 6px;
}

.warn-link {
  color: #7c5cfc;
  text-decoration: none;
  font-weight: 600;
}
.warn-link:hover { text-decoration: underline; }
</style>
