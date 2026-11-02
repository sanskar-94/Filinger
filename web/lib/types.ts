export interface Citation {
  paragraph_id: string;
  ticker: string | null;
  company: string | null;
  form: string;
  date_filed: string | null;
  section: string;
  accession: string;
  url: string;
  snippet: string;
}

export interface ToolCall {
  tool: string;
  input: Record<string, unknown>;
  ms?: number;
  error?: string | null;
}

export interface AgentAnswer {
  request_id: string;
  variant: string;
  experiment: string | null;
  answer: string;
  citations: Citation[];
  invalid_citations: string[];
  tool_calls: ToolCall[];
  mode: string;
  turns: number;
  latency_s: number;
  cost_usd: number;
  citation_valid_rate: number;
  error: string | null;
}

export interface Company {
  ticker: string;
  company: string;
  cik: number;
  filings: number;
  first: string;
  last: string;
}

export interface Paragraph {
  paragraph_id: string;
  text: string;
  ticker: string | null;
  company: string | null;
  form: string;
  section: string;
  accession: string;
  date_filed: string;
  url: string;
}

export interface FilingRow {
  accession: string;
  form: string;
  date_filed: string;
  items: string | null;
  car_3d: number | null;
}

export interface PricePoint {
  date: string;
  close: number;
}

export interface FilingDocument {
  accession: string;
  ticker: string | null;
  company: string | null;
  form: string;
  date_filed: string;
  url: string;
  sections: { section: string; paragraphs: { paragraph_id: string; para_idx: number; text: string }[] }[];
}

export interface ArmStats {
  users: number;
  exposures: number;
  rated: number;
  thumbs_up_rate: number;
  wilson_low: number;
  wilson_high: number;
  latency_p95_s: number;
  citation_valid_rate: number;
}

export interface Readout {
  experiment: string;
  control: string;
  treatment: string;
  decision: string;
  reasons: string[];
  srm: { users: Record<string, number>; p_value: number; mismatch: boolean };
  arms: Record<string, ArmStats>;
  primary: {
    diff: number;
    ci_clustered: [number, number];
    ci_newcombe: [number, number];
    relative_lift: number;
    p_value_clustered: number;
    msprt_p_final: number;
    msprt_confidence_sequence: [number, number];
    design_effect: number;
    mode: string;
  };
  guardrails: Record<string, { estimate: number; limit: number; breached: boolean; ci?: [number, number] }>;
  descriptions: Record<string, string>;
}

export interface ReviewItem {
  accession: string;
  form: string | null;
  company: string | null;
  paragraphs: { paragraph_id?: string; section: string; text: string }[];
  proposed: Labels;
  remaining: number;
  reviewed: number;
  total: number;
}

export interface Labels {
  guidance_direction: "raised" | "lowered" | "maintained" | "none";
  management_tone: "positive" | "neutral" | "negative";
  liquidity_concern: boolean;
  litigation_risk: boolean;
}

export interface ReviewStats {
  reviewed: number;
  skipped: number;
  corrected: number;
  total: number;
  fields: Record<string, { agreement: number | null; kappa: number | null }>;
}
