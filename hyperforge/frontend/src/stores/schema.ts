import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { fetchSchema } from '@/api/schema'
import type { AgentSchema, JsonSchema, SchemaRegistry, StageType } from '@/types/arag'
import { CONNECTABLE_KEYS } from '@/types/arag'

/**
 * Single source of truth for `/api/v1/ui/schema`.
 *
 * The backend embeds `$defs`, `connectable_keys` and `agent_module_to_def` in
 * the same payload, so no second request is needed.
 */
export const useSchemaStore = defineStore('schema', () => {
  const registry = ref<SchemaRegistry | null>(null)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function load() {
    loading.value = true
    error.value = null
    try {
      registry.value = await fetchSchema()
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  /** Merged `$defs` from the schema endpoint (for `$ref` resolution). */
  const defs = computed<Record<string, JsonSchema>>(() => registry.value?.$defs ?? {})

  /** Backend-declared connectable keys (with a sane fallback). */
  const connectableKeys = computed<readonly string[]>(
    () => registry.value?.connectable_keys ?? CONNECTABLE_KEYS,
  )

  /** Resolve a module id to its top-level `$defs` entry name. */
  function defForModule(moduleId: string): JsonSchema | undefined {
    const name = registry.value?.agent_module_to_def?.[moduleId]
    if (!name) return undefined
    return defs.value[name]
  }

  function getAgentSchema(stage: StageType, moduleId: string): AgentSchema | undefined {
    if (!registry.value) return undefined
    return registry.value[stage][moduleId]
  }

  function allAgentsForStage(stage: StageType): AgentSchema[] {
    if (!registry.value) return []
    return Object.values(registry.value[stage])
  }

  return {
    registry,
    loading,
    error,
    defs,
    connectableKeys,
    load,
    defForModule,
    getAgentSchema,
    allAgentsForStage,
  }
})
