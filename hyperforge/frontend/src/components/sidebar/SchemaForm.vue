<template>
  <div class="schema-form">
    <template v-for="(resolved, key) in visibleProperties" :key="key">
      <div class="form-field">
        <label class="form-label">
          {{ resolved.title || key }}
          <span v-if="isRequired(key)" class="required">*</span>
        </label>
        <span v-if="resolved.description" class="form-desc">{{ resolved.description }}</span>

        <!-- driver_select widget or source/sources field names -->
        <DriverSelect
          v-if="resolved.widget === 'driver_select' || isDriverSelectField(key)"
          :model-value="(fieldValue(key) as string | string[]) ?? (isArrayDriverField(key) ? [] : '')"
          :multiple="isArrayDriverField(key)"
          @update:model-value="(val) => emit('update', { [key]: val })"
        />

        <!-- Password / secret heuristics (only when no explicit widget) -->
        <input
          v-else-if="!resolved.widget && resolvedType(resolved) === 'string' && isSecret(key)"
          type="password"
          autocomplete="new-password"
          class="form-input"
          :value="String(fieldValue(key) ?? resolved.default ?? '')"
          :placeholder="resolved.description ?? ''"
          @input="emit('update', { [key]: ($event.target as HTMLInputElement).value })"
        />

        <!-- URL heuristic (only when no explicit widget and not a secret) -->
        <input
          v-else-if="!resolved.widget && resolvedType(resolved) === 'string' && isUrl(key)"
          type="url"
          class="form-input"
          :value="String(fieldValue(key) ?? resolved.default ?? '')"
          :placeholder="String(resolved.default ?? 'https://…')"
          @input="emit('update', { [key]: ($event.target as HTMLInputElement).value })"
        />

        <!-- Boolean toggle -->
        <div v-else-if="resolvedType(resolved) === 'boolean'" class="toggle-wrap">
          <label class="toggle">
            <input
              type="checkbox"
              :checked="Boolean(fieldValue(key))"
              @change="emit('update', { [key]: ($event.target as HTMLInputElement).checked })"
            />
            <span class="toggle-slider" />
          </label>
          <span class="toggle-label">{{ fieldValue(key) ? 'Yes' : 'No' }}</span>
        </div>

        <!-- Enum select -->
        <select
          v-else-if="resolved.enum"
          class="form-input"
          :value="fieldValue(key)"
          @change="emit('update', { [key]: ($event.target as HTMLSelectElement).value })"
        >
          <option v-for="opt in resolved.enum" :key="String(opt)" :value="String(opt)">
            {{ String(opt) }}
          </option>
        </select>

        <!-- model_select widget: text input + datalist -->
        <div v-else-if="resolved.widget === 'model_select'" class="model-field">
          <input
            type="text"
            class="form-input"
            list="model-datalist"
            :value="String(fieldValue(key) ?? resolved.default ?? '')"
            :placeholder="String(resolved.default ?? 'Select model…')"
            @input="emit('update', { [key]: ($event.target as HTMLInputElement).value })"
          />
          <datalist id="model-datalist">
            <option v-for="m in availableModels" :key="m" :value="m" />
          </datalist>
        </div>

        <!-- expandable_textarea widget -->
        <textarea
          v-else-if="resolved.widget === 'expandable_textarea'"
          class="form-input form-textarea form-textarea--expand"
          :value="String(fieldValue(key) ?? resolved.default ?? '')"
          :placeholder="resolved.description ?? ''"
          rows="3"
          @input="emit('update', { [key]: ($event.target as HTMLTextAreaElement).value })"
        />

        <!-- Number / integer -->
        <input
          v-else-if="resolvedType(resolved) === 'integer' || resolvedType(resolved) === 'number'"
          type="number"
          class="form-input"
          :value="fieldValue(key)"
          :placeholder="String(resolved.default ?? '')"
          @input="emit('update', { [key]: Number(($event.target as HTMLInputElement).value) })"
        />

        <!-- Array of strings → ArrayStringField -->
        <ArrayStringField
          v-else-if="resolvedType(resolved) === 'array' && resolvedType(resolved.items ?? {}) === 'string'"
          :model-value="(fieldValue(key) as string[]) ?? []"
          @update:model-value="(val) => emit('update', { [key]: val })"
        />

        <!-- Object with properties → recursive SchemaForm -->
        <div
          v-else-if="resolvedType(resolved) === 'object' && resolved.properties"
          class="nested-form"
        >
          <SchemaForm
            :schema="resolved"
            :value="(fieldValue(key) as Record<string, unknown>) ?? {}"
            :stage="stage"
            :defs="defs"
            @update="(updates) => emit('update', { [key]: { ...(fieldValue(key) as Record<string, unknown> ?? {}), ...updates } })"
          />
        </div>

        <!-- Object without properties → KeyValueField if additionalProperties, else JSON textarea -->
        <KeyValueField
          v-else-if="resolvedType(resolved) === 'object' && resolved.additionalProperties"
          :model-value="(fieldValue(key) as Record<string, string>) ?? {}"
          @update:model-value="(val) => emit('update', { [key]: val })"
        />

        <textarea
          v-else-if="resolvedType(resolved) === 'object'"
          class="form-input form-textarea"
          :value="JSON.stringify(fieldValue(key) ?? resolved.default ?? {}, null, 2)"
          rows="4"
          @blur="(e) => {
            try { emit('update', { [key]: JSON.parse((e.target as HTMLTextAreaElement).value) }) }
            catch { /* ignore invalid JSON */ }
          }"
        />

        <!-- Default: string -->
        <input
          v-else
          type="text"
          class="form-input"
          :value="String(fieldValue(key) ?? resolved.default ?? '')"
          :placeholder="String(resolved.default ?? '')"
          @input="emit('update', { [key]: ($event.target as HTMLInputElement).value })"
        />
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import type { JsonSchema, StageType } from '@/types/arag'
import { CONNECTABLE_KEYS, KNOWN_MODELS } from '@/types/arag'
import ArrayStringField from './ArrayStringField.vue'
import KeyValueField from './KeyValueField.vue'
import DriverSelect from '@/components/drivers/DriverSelect.vue'

// ── Props & emits ──────────────────────────────────────────────────────────

const props = defineProps<{
  schema: JsonSchema
  value: Record<string, unknown>
  /** Stage context (forwarded for nested forms / future use). */
  stage?: StageType
  /** Merged $defs map (from `useSchemaStore().defs` or local config_schema.$defs). */
  defs?: Record<string, JsonSchema>
}>()

const emit = defineEmits<{
  (e: 'update', updates: Record<string, unknown>): void
}>()

// ── Fields to skip entirely ────────────────────────────────────────────────
//
// Connectable keys are subagent links rendered as canvas children, not form
// fields. `module` is the discriminator (set elsewhere); `id` and `max_retries`
// are internal bookkeeping.
const SKIP_KEYS = new Set<string>([
  'module',
  'id',
  'max_retries',
  // Smart-agent auxiliary maps that are managed alongside the canvas children:
  'registered_agents_descriptions',
  'registered_agents_exposed_functions',
  ...CONNECTABLE_KEYS,
])

// ── Model datalist ─────────────────────────────────────────────────────────
const availableModels = ref<string[]>(KNOWN_MODELS)

onMounted(async () => {
  try {
    const res = await fetch('/api/v1/ui/models')
    if (res.ok) availableModels.value = await res.json()
  } catch {
    // silently fall back to the static list
  }
})

// ── $ref resolution ────────────────────────────────────────────────────────

function resolve(schema: JsonSchema): JsonSchema {
  if (schema.$ref) {
    const refKey = schema.$ref.replace('#/$defs/', '')
    const local = props.schema.$defs?.[refKey]
    const external = props.defs?.[refKey]
    const resolved = local ?? external
    if (resolved) return resolve(resolved)
  }
  if (schema.anyOf) {
    const nonNull = schema.anyOf.find((s) => s.type !== 'null' && !s.$ref?.includes('null'))
    if (nonNull) return resolve(nonNull)
  }
  return schema
}

// ── Driver select detection ────────────────────────────────────────────────

const DRIVER_SELECT_KEYS = new Set(['source', 'sources', 'kb'])

function isDriverSelectField(key: string): boolean {
  return DRIVER_SELECT_KEYS.has(key)
}

function isArrayDriverField(key: string): boolean {
  return key === 'sources'
}

// ── Field-name heuristics for password/url ─────────────────────────────────

const SECRET_PATTERN = /password|secret|api[_-]?key|token|credential|private/i
const URL_PATTERN = /url|endpoint|uri|base_url|host/i

function isSecret(key: string): boolean {
  return SECRET_PATTERN.test(key)
}

function isUrl(key: string): boolean {
  return URL_PATTERN.test(key) && !SECRET_PATTERN.test(key)
}

// ── Visible properties ─────────────────────────────────────────────────────

const visibleProperties = computed((): Record<string, JsonSchema> => {
  const raw = props.schema.properties ?? {}
  const out: Record<string, JsonSchema> = {}
  for (const [k, v] of Object.entries(raw)) {
    if (SKIP_KEYS.has(k)) continue
    const r = resolve(v)
    // Skip fields explicitly marked not_show, AND skip arrays-of-discriminated-
    // unions (those are subagent lists handled by the canvas).
    if (r.widget === 'not_show') continue
    if (r.type === 'array' && r.items && resolve(r.items).discriminator) continue
    if (r.discriminator) continue
    out[k] = r
  }
  return out
})

function resolvedType(schema: JsonSchema): string | null {
  if (Array.isArray(schema.type)) {
    return (schema.type as string[]).find((t) => t !== 'null') ?? null
  }
  return (schema.type as string | undefined) ?? null
}

function isRequired(key: string): boolean {
  return (props.schema.required ?? []).includes(key)
}

function fieldValue(key: string): unknown {
  return props.value[key]
}
</script>

<style scoped>
.schema-form { display: flex; flex-direction: column; gap: 14px; }

.form-field { display: flex; flex-direction: column; gap: 4px; }

.form-label {
  font-size: 12px; font-weight: 600; color: #bbb; text-transform: capitalize;
}
.required { color: #ff5555; margin-left: 2px; }

.form-desc { font-size: 11px; color: #666; line-height: 1.4; }

.form-input {
  background: #13131f; border: 1px solid #333; border-radius: 6px;
  padding: 7px 10px; color: #e0e0e0; font-size: 13px; outline: none;
  transition: border-color 0.15s; width: 100%; box-sizing: border-box;
}
.form-input:focus { border-color: #7c5cfc; }

.form-textarea { resize: vertical; font-family: monospace; font-size: 12px; line-height: 1.5; }
.form-textarea--expand { resize: both; min-height: 80px; max-height: 400px; }

select.form-input { cursor: pointer; }

.model-field { position: relative; }

.nested-form {
  border-left: 2px solid #2a2a40;
  padding-left: 12px;
  margin-top: 4px;
}

/* Toggle */
.toggle-wrap { display: flex; align-items: center; gap: 10px; }
.toggle { position: relative; display: inline-block; width: 40px; height: 22px; flex-shrink: 0; }
.toggle input { opacity: 0; width: 0; height: 0; position: absolute; }
.toggle-slider {
  position: absolute; inset: 0; background: #333;
  border-radius: 22px; transition: background 0.2s; cursor: pointer;
}
.toggle-slider::before {
  content: ''; position: absolute;
  width: 16px; height: 16px; border-radius: 50%; background: #fff;
  top: 3px; left: 3px; transition: transform 0.2s;
}
.toggle input:checked + .toggle-slider { background: #7c5cfc; }
.toggle input:checked + .toggle-slider::before { transform: translateX(18px); }
.toggle-label { font-size: 12px; color: #888; }
</style>
