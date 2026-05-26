// Mirrors the Python Pydantic models from hyperforge.standalone

export interface Rule {
  prompt: string;
}

export interface Rules {
  rules: Rule[];
}

export interface DriverConfig {
  id?: string | null;
  identifier: string;
  name: string;
  provider: string;
  config: Record<string, unknown>;
}

export interface WorkflowConfig {
  name: string;
  description?: string;
  parameters?: Record<string, unknown>;
  required: string[];
  rules: Rules;
  preprocess: AgentStepConfig[];
  context: AgentStepConfig[];
  generation: AgentStepConfig[];
  postprocess: AgentStepConfig[];
  /** UI-only: canvas positions per node id */
  _ui?: Record<string, {x: number; y: number}>;
}

/** Recursive agent step — includes subagent fields for conditional / ask / smart agents */
export interface AgentStepConfig {
  module: string;
  title?: string;
  source?: string;
  // Conditional agent branches (list[ContextAgentConfig])
  then?: AgentStepConfig[];
  else_?: AgentStepConfig[];
  // Ask-agent chaining (single ContextAgentConfig)
  fallback?: AgentStepConfig;
  next_agent?: AgentStepConfig;
  // Smart agent / orchestrator children
  registered_agents?: AgentStepConfig[];
  agents?: AgentStepConfig[];
  [key: string]: unknown;
}

/** Keys whose values are subagents — rendered as canvas children, not form fields. */
export const CONNECTABLE_KEYS = [
  "fallback",
  "next_agent",
  "then",
  "else_",
  "agents",
  "registered_agents",
] as const;
export type ConnectableKey = (typeof CONNECTABLE_KEYS)[number];

/** Keys whose value is a *single* subagent (vs. a list). */
export const SINGLE_CONNECTABLE_KEYS: ReadonlyArray<ConnectableKey> = [
  "fallback",
  "next_agent",
];

export interface PromptArgument {
  name: string;
  description?: string;
  required?: boolean;
}

export interface PromptConfig {
  name: string;
  description: string;
  prompt: string;
  arguments?: PromptArgument[];
  prompt_id?: string;
}

export interface AgentConfig {
  title?: string;
  description?: string;
  instructions?: string;
  drivers: DriverConfig[];
  rules: Rules;
  workflows: Record<string, WorkflowConfig>;
  prompts: PromptConfig[];
}

export type StandaloneConfig = Record<string, AgentConfig>;

// ── Schema registry ───────────────────────────────────────────────────────

export interface AgentSchema {
  id: string;
  agent_type: "preprocess" | "context" | "generation" | "postprocess";
  title: string;
  description: string;
  config_schema: JsonSchema;
}

export interface DriverSchema {
  id: string;
  title: string;
  description: string;
  config_schema: JsonSchema;
}

export interface SchemaRegistry {
  preprocess: Record<string, AgentSchema>;
  context: Record<string, AgentSchema>;
  generation: Record<string, AgentSchema>;
  postprocess: Record<string, AgentSchema>;
  drivers: Record<string, DriverSchema>;
  /** Merged $defs from all agent/driver schemas (for $ref resolution). */
  $defs?: Record<string, JsonSchema>;
  /** Mirrors backend `CONNECTABLE_KEYS`. */
  connectable_keys?: string[];
  /** Map module_id → $defs key (top-level config-schema title). */
  agent_module_to_def?: Record<string, string>;
}

// ── JSON Schema (subset) ──────────────────────────────────────────────────

export interface JsonSchema {
  type?: string | string[];
  title?: string;
  description?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  items?: JsonSchema;
  enum?: unknown[];
  default?: unknown;
  anyOf?: JsonSchema[];
  allOf?: JsonSchema[];
  oneOf?: JsonSchema[];
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  additionalProperties?: boolean | JsonSchema;
  // Custom widget hints from json_schema_extra
  widget?: string;
  show_in_node?: boolean;
  discriminator?: {propertyName: string; mapping?: Record<string, string>};
}

// ── Streaming chat ────────────────────────────────────────────────────────

/** Mirrors `hyperforge.api.models.InteractionOperation`. */
export enum InteractionOperation {
  QUESTION = 0,
  QUIT = 1,
}

export enum AnswerOperation {
  ANSWER = 0,
  START = 2,
  DONE = 3,
  ERROR = 4,
  AGENT_REQUEST = 5,
}

export interface AragChunk {
  text: string;
  score?: number;
}

export interface AragContext {
  chunks: AragChunk[];
  source: string;
}

export interface AragStep {
  module: string;
  title?: string;
}

/** Mirrors `hyperforge.interaction.ARAGException`. */
export interface ARAGException {
  detail: string;
  extra?: Record<string, unknown>;
}

/** Mirrors `hyperforge.interaction.OAuthAuthenticateURL`. */
export interface OAuthAuthenticateURL {
  oauth_url: string;
}

/** Feedback request sent by an agent over WebSocket when it needs user input */
export interface Feedback {
  request_id: string;
  feedback_id: string;
  question: string;
  module: string;
  agent_id: string;
  data?: unknown;
  timeout_ms: number;
  response_schema?: JsonSchema | null;
  get_credentials?: Record<string, unknown> | null;
  credentials?: Record<string, Record<string, unknown>> | null;
}

/** Visualization payload from the agent (subset — frontend just forwards/displays). */
export interface Visualization {
  [key: string]: unknown;
}

export interface AragAnswer {
  operation?: AnswerOperation;
  answer?: string;
  answer_citations?: Record<string, unknown>;
  answer_urls?: string[];
  agent_request?: string;
  generated_text?: string;
  context?: AragContext;
  step?: AragStep;
  possible_answer?: {answer: string; module: string};
  exception?: ARAGException;
  feedback?: Feedback;
  oauth?: OAuthAuthenticateURL;
  seqid?: number;
  original_question_uuid?: string;
  actual_question_uuid?: string;
  data_visualizations?: Visualization[];
}

/** Client reply to an AGENT_REQUEST */
export interface UserToAgentInteraction {
  op: "user_response";
  request_id: string;
  response: string;
}

export type StageType = "preprocess" | "context" | "generation" | "postprocess";

export const STAGE_ORDER: StageType[] = [
  "preprocess",
  "context",
  "generation",
  "postprocess",
];

export const STAGE_COLORS: Record<StageType, string> = {
  preprocess: "#7c5cfc",
  context: "#2196f3",
  generation: "#4caf50",
  postprocess: "#ff9800",
};

export const STAGE_LABELS: Record<StageType, string> = {
  preprocess: "Preprocess",
  context: "Context",
  generation: "Generation",
  postprocess: "Postprocess",
};

/** Known model IDs for the model_select datalist */
export const KNOWN_MODELS: string[] = [
  "chatgpt-azure-4o-mini",
  "chatgpt-azure-4o",
  "chatgpt4o",
  "chatgpt-4.1",
  "chatgpt-5",
  "chatgpt-o3-mini",
  "gemini-2.5-flash",
  "gemini-2.5-flash-lite",
  "gemini-3-flash-preview",
  "claude-4-5-haiku",
  "claude-4-5-sonnet",
  "gcp-claude-4-5-haiku",
  "gcp-claude-4-5-sonnet",
];
