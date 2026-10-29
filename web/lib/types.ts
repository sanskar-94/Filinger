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
