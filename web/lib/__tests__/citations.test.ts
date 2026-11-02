import { describe, expect, it } from "vitest";
import { formatPct, parseAnswer } from "../citations";

const A = "aaaaaaaaaaaaaaaa";
const B = "bbbbbbbbbbbbbbbb";

describe("parseAnswer", () => {
  it("numbers citations by first appearance and reuses numbers", () => {
    const segs = parseAnswer(`Raised [cite:${A}] then cut [cite:${B}] again [cite:${A}].`, [A, B]);
    const cites = segs.filter((s) => s.kind === "cite");
    expect(cites.map((c) => (c.kind === "cite" ? c.n : 0))).toEqual([1, 2, 1]);
    expect(segs[0]).toEqual({ kind: "text", value: "Raised " });
    expect(segs.at(-1)).toEqual({ kind: "text", value: "." });
  });

  it("flags citations the backend did not validate", () => {
    const segs = parseAnswer(`x [cite:${A}]`, []);
    expect(segs[1]).toMatchObject({ kind: "cite", valid: false });
  });

  it("leaves text without citations untouched", () => {
    expect(parseAnswer("no citations here", [])).toEqual([{ kind: "text", value: "no citations here" }]);
  });
});

describe("formatPct", () => {
  it("formats signed percentages", () => {
    expect(formatPct(0.0123)).toBe("+1.23%");
    expect(formatPct(-0.5, 1)).toBe("-50.0%");
    expect(formatPct(null)).toBe("n/a");
  });
});

describe("sectionLabel and axisPct", () => {
  it("names known sections and humanises unknown ones", async () => {
    const { sectionLabel } = await import("../citations");
    expect(sectionLabel("mdna")).toBe("Management's discussion and analysis");
    expect(sectionLabel("custom_part")).toBe("custom part");
  });
  it("maps values onto an axis and clamps", async () => {
    const { axisPct } = await import("../citations");
    expect(axisPct(0, -5, 5)).toBe(50);
    expect(axisPct(10, -5, 5)).toBe(100);
    expect(axisPct(1, 1, 1)).toBe(50);
  });
});
