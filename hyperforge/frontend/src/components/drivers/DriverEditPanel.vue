<template>
  <div class="panel-backdrop" @click.self="emit('close')">
    <div class="edit-panel">
      <div class="panel-header">
        <h2 class="panel-title">{{ isNew ? 'Add driver' : 'Edit driver' }}</h2>
        <button class="close-btn" @click="emit('close')">✕</button>
      </div>

      <div class="panel-body">
        <!-- Agent selector (only shown on add) -->
        <div v-if="isNew" class="form-field">
          <label class="form-label">Agent <span class="required">*</span></label>
          <select class="form-input" v-model="localAgentId">
            <option v-for="id in workflowStore.agentIds" :key="id" :value="id">{{ id }}</option>
          </select>
        </div>

        <!-- Provider -->
        <div class="form-field">
          <label class="form-label">Provider <span class="required">*</span></label>
          <select class="form-input" v-model="localProvider" :disabled="!isNew">
            <option value="">Select provider…</option>
            <option v-for="(ds, id) in driverSchemas" :key="id" :value="id">
              {{ ds.title || id }}
            </option>
          </select>
          <span v-if="selectedDriverSchema" class="form-desc">{{ selectedDriverSchema.description }}</span>
        </div>

        <!-- Name -->
        <div class="form-field">
          <label class="form-label">Name <span class="required">*</span></label>
          <input
            class="form-input"
            v-model="localName"
            placeholder="My NucliaDB KB"
            @input="syncIdentifier"
          />
        </div>

        <!-- Identifier -->
        <div class="form-field">
          <label class="form-label">
            Identifier
            <span class="required">*</span>
            <span class="form-desc-inline"> (used in agent config as reference key)</span>
          </label>
          <input
            class="form-input"
            v-model="localIdentifier"
            placeholder="my-nucliadb-kb"
            spellcheck="false"
          />
          <span v-if="identifierConflict" class="form-error">
            Identifier already in use. Choose another.
          </span>
        </div>

        <!-- Config fields from driver schema -->
        <div v-if="selectedDriverSchema && configSchema" class="config-section">
          <h4 class="section-title">Configuration</h4>
          <SchemaForm
            :schema="configSchema"
            :value="localConfig"
            :defs="combinedDefs"
            @update="(updates) => (localConfig = { ...localConfig, ...updates })"
          />
        </div>

        <div v-else-if="localProvider && !selectedDriverSchema" class="no-schema">
          No schema found for provider <code>{{ localProvider }}</code>.
        </div>
      </div>

      <div class="panel-footer">
        <button class="btn-ghost" @click="emit('close')">Cancel</button>
        <button class="btn-primary" :disabled="!canSave" @click="onSave">
          {{ isNew ? 'Add driver' : 'Save changes' }}
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useSchemaStore } from '@/stores/schema'
import { useWorkflowStore } from '@/stores/workflow'
import type { DriverConfig, JsonSchema } from '@/types/arag'
import SchemaForm from '@/components/sidebar/SchemaForm.vue'

// ── Props & emits ──────────────────────────────────────────────────────────

const props = defineProps<{
  /** null = adding a new driver */
  driver: DriverConfig | null
  agentId: string
}>()

const emit = defineEmits<{
  (e: 'save', payload: { agentId: string; driver: DriverConfig; isNew: boolean }): void
  (e: 'close'): void
}>()

const schemaStore = useSchemaStore()
const workflowStore = useWorkflowStore()

const isNew = computed(() => props.driver === null)

// ── Local state ────────────────────────────────────────────────────────────

const localAgentId = ref(props.agentId)
const localProvider = ref(props.driver?.provider ?? '')
const localName = ref(props.driver?.name ?? '')
const localIdentifier = ref(props.driver?.identifier ?? '')
const localConfig = ref<Record<string, unknown>>(
  (props.driver?.config as Record<string, unknown>) ?? {},
)

// Track whether user manually edited the identifier
let identifierManuallyEdited = !isNew.value

// ── Schema helpers ─────────────────────────────────────────────────────────

const driverSchemas = computed(() => schemaStore.registry?.drivers ?? {})

const selectedDriverSchema = computed(() =>
  localProvider.value ? driverSchemas.value[localProvider.value] : null,
)

/** Resolve the config $ref from the driver schema */
const configSchema = computed(() => {
  const ds = selectedDriverSchema.value
  if (!ds) return null
  const cs = ds.config_schema
  if (!cs) return null
  // The config_schema top-level may itself be a $ref to a $defs entry
  if (cs.$ref) {
    const key = cs.$ref.replace('#/$defs/', '')
    return cs.$defs?.[key] ?? null
  }
  // Or it may have a `config` property that is a $ref
  const configProp = cs.properties?.config
  if (configProp?.$ref) {
    const key = configProp.$ref.replace('#/$defs/', '')
    return cs.$defs?.[key] ?? null
  }
  // Otherwise return the schema as-is
  return cs
})

// Reset config when provider changes — seed from each top-level property's
// `default` so users see the schema-suggested values rather than an empty form.
watch(localProvider, () => {
  if (!isNew.value) return
  const cs = configSchema.value
  if (!cs?.properties) {
    localConfig.value = {}
    return
  }
  const seeded: Record<string, unknown> = {}
  for (const [key, prop] of Object.entries(cs.properties)) {
    if (prop?.default !== undefined) seeded[key] = prop.default
  }
  localConfig.value = seeded
})

/** Merge driver-local $defs with the global registry $defs. */
const combinedDefs = computed<Record<string, JsonSchema>>(() => ({
  ...(selectedDriverSchema.value?.config_schema?.$defs ?? {}),
  ...schemaStore.defs,
}))

// ── Identifier auto-generation ─────────────────────────────────────────────

function slugify(s: string): string {
  return s
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

function syncIdentifier() {
  if (!identifierManuallyEdited) {
    localIdentifier.value = slugify(localName.value)
  }
}

// Allow user to override the auto-generated identifier
watch(localIdentifier, (val, old) => {
  if (val !== slugify(localName.value) && val !== old) {
    identifierManuallyEdited = true
  }
})

// Pre-populate suggestions from existing driver identifiers (for add mode)
const existingIdentifiers = computed(() =>
  workflowStore.allDrivers.map((d) => d.identifier),
)

const identifierConflict = computed(() => {
  if (!localIdentifier.value) return false
  if (!isNew.value && localIdentifier.value === props.driver?.identifier) return false
  return existingIdentifiers.value.includes(localIdentifier.value)
})

// ── Save ───────────────────────────────────────────────────────────────────

const canSave = computed(() => {
  return (
    localProvider.value &&
    localName.value.trim() &&
    localIdentifier.value.trim() &&
    !identifierConflict.value
  )
})

function onSave() {
  if (!canSave.value) return
  emit('save', {
    agentId: localAgentId.value,
    isNew: isNew.value,
    driver: {
      id: props.driver?.id ?? null,
      identifier: localIdentifier.value.trim(),
      name: localName.value.trim(),
      provider: localProvider.value,
      config: localConfig.value,
    },
  })
}
</script>

<style scoped>
.panel-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.5);
  z-index: 150;
  display: flex;
  justify-content: flex-end;
}

.edit-panel {
  width: 440px;
  max-width: 95vw;
  background: #1a1a2e;
  border-left: 1px solid #2a2a40;
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
  animation: slide-in 0.2s ease;
}

@keyframes slide-in {
  from { transform: translateX(100%); }
  to   { transform: translateX(0); }
}

.panel-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px 20px;
  border-bottom: 1px solid #2a2a40;
  flex-shrink: 0;
}

.panel-title {
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
  flex-shrink: 0;
}
.close-btn:hover { color: #ccc; }

.panel-body {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.panel-footer {
  display: flex;
  gap: 8px;
  justify-content: flex-end;
  padding: 14px 20px;
  border-top: 1px solid #2a2a40;
  flex-shrink: 0;
}

.form-field { display: flex; flex-direction: column; gap: 4px; }

.form-label {
  font-size: 12px; font-weight: 600; color: #bbb; text-transform: capitalize;
}
.required { color: #ff5555; margin-left: 2px; }
.form-desc { font-size: 11px; color: #666; line-height: 1.4; }
.form-desc-inline { font-size: 11px; color: #555; font-weight: 400; text-transform: none; }

.form-error {
  font-size: 11px;
  color: #ff5555;
}

.form-input {
  background: #13131f;
  border: 1px solid #333;
  border-radius: 6px;
  padding: 7px 10px;
  color: #e0e0e0;
  font-size: 13px;
  outline: none;
  transition: border-color 0.15s;
  width: 100%;
  box-sizing: border-box;
}
.form-input:focus { border-color: #7c5cfc; }
.form-input:disabled { opacity: 0.5; cursor: not-allowed; }

select.form-input { cursor: pointer; }

.config-section { display: flex; flex-direction: column; gap: 10px; }

.section-title {
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: #555;
  margin: 0;
  padding-top: 8px;
  border-top: 1px solid #1e1e30;
}

.no-schema {
  font-size: 13px;
  color: #666;
}
.no-schema code {
  font-family: monospace;
  background: #252535;
  padding: 1px 5px;
  border-radius: 4px;
  font-size: 12px;
}

.btn-primary {
  background: #7c5cfc;
  border: none;
  color: #fff;
  padding: 8px 16px;
  border-radius: 8px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-primary:hover:not(:disabled) { background: #6a4de8; }
.btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

.btn-ghost {
  background: none;
  border: 1px solid #333;
  color: #aaa;
  padding: 7px 14px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  transition: border-color 0.15s;
}
.btn-ghost:hover { border-color: #666; }
</style>
