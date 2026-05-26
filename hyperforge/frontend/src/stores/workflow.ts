import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { v4 as uuidv4 } from 'uuid'
import { fetchConfig, saveConfig } from '@/api/config'
import type {
  StandaloneConfig,
  AgentConfig,
  WorkflowConfig,
  AgentStepConfig,
  StageType,
  DriverConfig,
  ConnectableKey,
} from '@/types/arag'
import { CONNECTABLE_KEYS, SINGLE_CONNECTABLE_KEYS, STAGE_ORDER } from '@/types/arag'

// ── Node path helpers ──────────────────────────────────────────────────────
//
// Node IDs encode the full path to a step in the (possibly nested) config:
//
//   Top-level:                   "context:0"
//   List subagent (then/else_):  "context:0:then:1"
//   Single subagent (fallback):  "context:0:fallback:*"
//   Smart agent registered:      "context:0:registered_agents:2"
//
// Segment pattern: (<stage|connectable_key>:<index|"*">)+
//   - `*` is a sentinel meaning "single value, no list index"
//     (used for `fallback` and `next_agent`).

export type NodePath = string

const SINGLE_INDEX = '*'

function isListKey(key: string): boolean {
  return !(SINGLE_CONNECTABLE_KEYS as readonly string[]).includes(key)
}

/** Parse a node path into a sequence of (key, index) pairs. `*` for single. */
function parsePath(nodeId: NodePath): Array<{ key: string; idx: number | typeof SINGLE_INDEX }> {
  const segments: Array<{ key: string; idx: number | typeof SINGLE_INDEX }> = []
  const parts = nodeId.split(':')
  for (let i = 0; i < parts.length; i += 2) {
    const rawIdx = parts[i + 1]
    segments.push({
      key: parts[i],
      idx: rawIdx === SINGLE_INDEX ? SINGLE_INDEX : parseInt(rawIdx, 10),
    })
  }
  return segments
}

/** Navigate a workflow to a step using a node path. Returns the config or null. */
export function getStepByPath(wf: WorkflowConfig, nodeId: NodePath): AgentStepConfig | null {
  const segs = parsePath(nodeId)
  if (segs.length === 0) return null

  const [first, ...rest] = segs
  // Top-level segment is always a stage (list).
  let current: AgentStepConfig | null =
    (wf[first.key as StageType] as AgentStepConfig[])?.[first.idx as number] ?? null
  if (!current) return null

  for (const seg of rest) {
    if (seg.idx === SINGLE_INDEX) {
      current = (current[seg.key] as AgentStepConfig | undefined) ?? null
    } else {
      const arr = current[seg.key] as AgentStepConfig[] | undefined
      current = arr?.[seg.idx as number] ?? null
    }
    if (!current) return null
  }
  return current
}

/**
 * Mutate a step in-place using a node path. The mutator receives the current
 * step plus a "container" (parent array + index, or parent step + key for
 * single-valued links) so list and single mutations share an interface.
 */
type Mutator = (
  step: AgentStepConfig,
  container:
    | { kind: 'list'; arr: AgentStepConfig[]; idx: number }
    | { kind: 'single'; parent: AgentStepConfig; key: string },
) => void

function mutateStepByPath(wf: WorkflowConfig, nodeId: NodePath, mutate: Mutator): void {
  const segs = parsePath(nodeId)
  if (segs.length === 0) return

  const [first, ...rest] = segs
  const topArr = wf[first.key as StageType] as AgentStepConfig[]
  if (!topArr) return

  if (rest.length === 0) {
    mutate(topArr[first.idx as number], { kind: 'list', arr: topArr, idx: first.idx as number })
    return
  }

  let current = topArr[first.idx as number]
  for (let i = 0; i < rest.length - 1; i++) {
    const seg = rest[i]
    if (seg.idx === SINGLE_INDEX) {
      const next = current[seg.key] as AgentStepConfig | undefined
      if (!next) return
      current = next
    } else {
      const arr = current[seg.key] as AgentStepConfig[] | undefined
      if (!arr) return
      const next = arr[seg.idx as number]
      if (!next) return
      current = next
    }
  }

  const lastSeg = rest[rest.length - 1]
  if (lastSeg.idx === SINGLE_INDEX) {
    const step = current[lastSeg.key] as AgentStepConfig | undefined
    if (!step) return
    mutate(step, { kind: 'single', parent: current, key: lastSeg.key })
  } else {
    const lastArr = current[lastSeg.key] as AgentStepConfig[] | undefined
    if (!lastArr) return
    mutate(lastArr[lastSeg.idx as number], {
      kind: 'list',
      arr: lastArr,
      idx: lastSeg.idx as number,
    })
  }
}

/** Get the parent stage of a (possibly nested) node path */
export function pathToStage(nodeId: NodePath): StageType {
  return nodeId.split(':')[0] as StageType
}

// ── Canvas node data ───────────────────────────────────────────────────────

export interface CanvasNodeData {
  id: NodePath
  stage: StageType
  stepIndex: number
  agentId: string
  workflowId: string
  config: AgentStepConfig
  /** Connectable key linking this node to its parent (e.g. "then", "registered_agents"). */
  parentLinkType?: ConnectableKey
  /** Parent node ID (for subagents) */
  parentId?: NodePath
}

// ── Store ──────────────────────────────────────────────────────────────────

export const useWorkflowStore = defineStore('workflow', () => {
  const config = ref<StandaloneConfig>({})
  const loading = ref(false)
  const saving = ref(false)
  const dirty = ref(false)
  const error = ref<string | null>(null)

  const activeAgentId = ref<string | null>(null)
  const activeWorkflowId = ref<string>('default')
  const selectedNodeId = ref<string | null>(null)

  // ── Getters ──────────────────────────────────────────────────────────────

  const agentIds = computed(() => Object.keys(config.value))

  const activeAgent = computed((): AgentConfig | null => {
    if (!activeAgentId.value) return null
    return config.value[activeAgentId.value] ?? null
  })

  const activeWorkflow = computed((): WorkflowConfig | null => {
    if (!activeAgent.value) return null
    return activeAgent.value.workflows[activeWorkflowId.value] ?? null
  })

  const canvasNodes = computed((): CanvasNodeData[] => {
    if (!activeWorkflow.value || !activeAgentId.value) return []
    const nodes: CanvasNodeData[] = []
    _collectNodes(
      activeWorkflow.value,
      activeAgentId.value,
      activeWorkflowId.value,
      nodes,
    )
    return nodes
  })

  // ── Actions ──────────────────────────────────────────────────────────────

  async function load() {
    loading.value = true
    error.value = null
    try {
      config.value = await fetchConfig()
      const first = Object.keys(config.value)[0]
      if (first) activeAgentId.value = first
      dirty.value = false
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e)
    } finally {
      loading.value = false
    }
  }

  async function save() {
    saving.value = true
    error.value = null
    try {
      const stripped = stripUiMeta(config.value)
      await saveConfig(stripped)
      dirty.value = false
    } catch (e: unknown) {
      error.value = e instanceof Error ? e.message : String(e)
      throw e
    } finally {
      saving.value = false
    }
  }

  function setActiveAgent(agentId: string) {
    activeAgentId.value = agentId
    activeWorkflowId.value = 'default'
    selectedNodeId.value = null
  }

  function setActiveWorkflow(workflowId: string) {
    activeWorkflowId.value = workflowId
    selectedNodeId.value = null
  }

  function selectNode(nodeId: string | null) {
    selectedNodeId.value = nodeId
  }

  /** Return the AgentStepConfig for any node path (including subagents). */
  function getNodeConfig(nodeId: NodePath): AgentStepConfig | null {
    if (!activeWorkflow.value) return null
    return getStepByPath(activeWorkflow.value, nodeId)
  }

  /** Update the config for a node at any path depth. */
  function updateNodeConfig(nodeId: NodePath, updates: Partial<AgentStepConfig>) {
    if (!activeAgentId.value || !activeWorkflow.value) return
    const wf = config.value[activeAgentId.value].workflows[activeWorkflowId.value]
    mutateStepByPath(wf, nodeId, (step, container) => {
      const merged = { ...step, ...updates }
      if (container.kind === 'list') {
        container.arr[container.idx] = merged
      } else {
        container.parent[container.key] = merged
      }
    })
    dirty.value = true
  }

  /** Add a top-level step to a stage. */
  function addStep(stage: StageType, step: AgentStepConfig) {
    if (!activeAgentId.value) return
    const wf = config.value[activeAgentId.value].workflows[activeWorkflowId.value]
    if (!wf[stage]) wf[stage] = []
    wf[stage].push({ ...step })
    dirty.value = true
  }

  /**
   * Add a subagent under an existing node, linked by `linkType`.
   * For list keys (then/else_/agents/registered_agents) appends to the array.
   * For single keys (fallback/next_agent) overwrites the existing value.
   * Returns the new child's NodePath.
   */
  function addChild(
    parentNodeId: NodePath,
    linkType: ConnectableKey,
    moduleId: string,
  ): NodePath | null {
    if (!activeAgentId.value || !activeWorkflow.value) return null
    const wf = config.value[activeAgentId.value].workflows[activeWorkflowId.value]
    let newPath: NodePath | null = null
    mutateStepByPath(wf, parentNodeId, (step) => {
      const newChild: AgentStepConfig = { module: moduleId }
      if (isListKey(linkType)) {
        const existing = (step[linkType] as AgentStepConfig[] | undefined) ?? []
        existing.push(newChild)
        ;(step as Record<string, unknown>)[linkType] = existing
        newPath = `${parentNodeId}:${linkType}:${existing.length - 1}`
      } else {
        ;(step as Record<string, unknown>)[linkType] = newChild
        newPath = `${parentNodeId}:${linkType}:${SINGLE_INDEX}`
      }
    })
    if (newPath) dirty.value = true
    return newPath
  }

  /** Remove a child node (subagent) by path. Equivalent to removeStep, kept for clarity. */
  function removeChild(nodeId: NodePath) {
    removeStep(nodeId)
  }

  /** Remove a step at any depth by node path. */
  function removeStep(nodeId: NodePath) {
    if (!activeAgentId.value) return
    const wf = config.value[activeAgentId.value].workflows[activeWorkflowId.value]
    mutateStepByPath(wf, nodeId, (_step, container) => {
      if (container.kind === 'list') {
        container.arr.splice(container.idx, 1)
      } else {
        delete container.parent[container.key]
      }
    })
    if (selectedNodeId.value === nodeId) selectedNodeId.value = null
    dirty.value = true
  }

  /** Persist canvas node positions into _ui metadata. */
  function updateNodePosition(nodeId: string, x: number, y: number) {
    if (!activeAgentId.value) return
    const wf = config.value[activeAgentId.value].workflows[activeWorkflowId.value]
    if (!wf._ui) wf._ui = {}
    wf._ui[nodeId] = { x, y }
  }

  /** Update top-level agent metadata. */
  function updateAgentMeta(
    agentId: string,
    updates: { title?: string; description?: string; instructions?: string },
  ) {
    if (!config.value[agentId]) return
    Object.assign(config.value[agentId], updates)
    dirty.value = true
  }

  // ── Driver CRUD ───────────────────────────────────────────────────────────

  /** All drivers across all agents as flat list with agentId attached. */
  const allDrivers = computed((): Array<DriverConfig & { agentId: string }> => {
    const out: Array<DriverConfig & { agentId: string }> = []
    for (const [agentId, agent] of Object.entries(config.value)) {
      for (const d of agent.drivers ?? []) {
        out.push({ ...d, agentId })
      }
    }
    return out
  })

  function addDriver(agentId: string, driver: DriverConfig) {
    if (!config.value[agentId]) return
    if (!config.value[agentId].drivers) config.value[agentId].drivers = []
    config.value[agentId].drivers.push({ ...driver })
    dirty.value = true
  }

  function updateDriver(agentId: string, identifier: string, updates: Partial<DriverConfig>) {
    const agent = config.value[agentId]
    if (!agent) return
    const idx = (agent.drivers ?? []).findIndex((d) => d.identifier === identifier)
    if (idx === -1) return
    agent.drivers[idx] = { ...agent.drivers[idx], ...updates }
    dirty.value = true
  }

  function removeDriver(agentId: string, identifier: string) {
    const agent = config.value[agentId]
    if (!agent) return
    agent.drivers = (agent.drivers ?? []).filter((d) => d.identifier !== identifier)
    dirty.value = true
  }

  return {
    config,
    loading,
    saving,
    dirty,
    error,
    activeAgentId,
    activeWorkflowId,
    selectedNodeId,
    agentIds,
    activeAgent,
    activeWorkflow,
    canvasNodes,
    load,
    save,
    setActiveAgent,
    setActiveWorkflow,
    selectNode,
    getNodeConfig,
    updateNodeConfig,
    addStep,
    addChild,
    removeStep,
    removeChild,
    updateNodePosition,
    updateAgentMeta,
    allDrivers,
    addDriver,
    updateDriver,
    removeDriver,
  }
})

// ---------------------------------------------------------------------------
// Recursive node collection
// ---------------------------------------------------------------------------

function _collectNodes(
  wf: WorkflowConfig,
  agentId: string,
  workflowId: string,
  out: CanvasNodeData[],
): void {
  for (const stage of STAGE_ORDER) {
    const steps = (wf[stage] as AgentStepConfig[]) ?? []
    steps.forEach((step, idx) => {
      const nodeId = `${stage}:${idx}`
      out.push({
        id: nodeId,
        stage,
        stepIndex: idx,
        agentId,
        workflowId,
        config: step,
      })
      _collectSubagents(step, nodeId, stage, agentId, workflowId, out)
    })
  }
}

function _collectSubagents(
  step: AgentStepConfig,
  parentPath: NodePath,
  stage: StageType,
  agentId: string,
  workflowId: string,
  out: CanvasNodeData[],
): void {
  for (const key of CONNECTABLE_KEYS) {
    const value = step[key]
    if (!value) continue

    // Subagents that change stage when traversed (smart agent's children run
    // in the context stage regardless of the parent's stage).
    const childStage: StageType =
      key === 'registered_agents' || key === 'agents' ? 'context' : stage

    if (Array.isArray(value)) {
      ;(value as AgentStepConfig[]).forEach((sub, idx) => {
        const nodeId = `${parentPath}:${key}:${idx}`
        out.push({
          id: nodeId,
          stage: childStage,
          stepIndex: idx,
          agentId,
          workflowId,
          config: sub,
          parentLinkType: key,
          parentId: parentPath,
        })
        _collectSubagents(sub, nodeId, childStage, agentId, workflowId, out)
      })
    } else {
      const sub = value as AgentStepConfig
      const nodeId = `${parentPath}:${key}:${SINGLE_INDEX}`
      out.push({
        id: nodeId,
        stage: childStage,
        stepIndex: 0,
        agentId,
        workflowId,
        config: sub,
        parentLinkType: key,
        parentId: parentPath,
      })
      _collectSubagents(sub, nodeId, childStage, agentId, workflowId, out)
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function stripUiMeta(cfg: StandaloneConfig): StandaloneConfig {
  const out: StandaloneConfig = {}
  for (const [agentId, agentCfg] of Object.entries(cfg)) {
    const workflows: Record<string, WorkflowConfig> = {}
    for (const [wfId, wf] of Object.entries(agentCfg.workflows)) {
      const { _ui: _ignored, ...rest } = wf as WorkflowConfig & { _ui?: unknown }
      workflows[wfId] = rest as WorkflowConfig
    }
    out[agentId] = { ...agentCfg, workflows }
  }
  return out
}

void uuidv4
