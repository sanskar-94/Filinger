export type Segment =
  | { kind: "text"; value: string }
  | { kind: "cite"; id: string; n: number; valid: boolean };

const CITE = /\[cite:([0-9a-f]{16})\]/g;

/**
 * Split an answer into text and citation segments. Citations are numbered in order of first appearance,
 * so a paragraph cited twice keeps the same number. Ids that the backend marked invalid stay visible but
 * are flagged, instead of silently disappearing.
 */
export function parseAnswer(text: string, validIds: Iterable<string>): Segment[] {
  const valid = new Set(validIds);
  const numbers = new Map<string, number>();
  const out: Segment[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    const id = m[1] ?? "";
    const start = m.index ?? 0;
    if (start > last) out.push({ kind: "text", value: text.slice(last, start) });
    if (!numbers.has(id)) numbers.set(id, numbers.size + 1);
    out.push({ kind: "cite", id, n: numbers.get(id) ?? 0, valid: valid.has(id) });
    last = start + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", value: text.slice(last) });
  return out;
}

export function formatPct(x: number | null | undefined, digits = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return "n/a";
  const v = x * 100;
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`;
}

export function sectionLabel(section: string): string {
  const names: Record<string, string> = {
    mdna: "Management's discussion and analysis",
    risk_factors: "Risk factors",
    legal: "Legal proceedings",
    "8k_results_of_operations": "Results of operations (8-K item 2.02)",
    "8k_material_agreement": "Material agreement (8-K item 1.01)",
    "8k_officer_change": "Officer change (8-K item 5.02)",
    "8k_other_events": "Other events (8-K item 8.01)",
    "8k_exhibits": "Exhibits (8-K item 9.01)",
    full_text: "Full text",
  };
  return names[section] ?? section.replaceAll("_", " ");
}

/** Position of a value on a [lo, hi] axis as a percentage, clamped, for simple interval bars. */
export function axisPct(v: number, lo: number, hi: number): number {
  if (!(hi > lo)) return 50;
  return Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
}
