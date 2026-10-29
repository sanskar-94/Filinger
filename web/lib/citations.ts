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
