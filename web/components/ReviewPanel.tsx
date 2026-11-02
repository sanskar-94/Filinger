"use client";

import { useCallback, useEffect, useState } from "react";
import { sectionLabel } from "@/lib/citations";
import type { Labels, ReviewItem, ReviewStats } from "@/lib/types";

const GUIDANCE: Labels["guidance_direction"][] = ["raised", "lowered", "maintained", "none"];
const TONE: Labels["management_tone"][] = ["positive", "neutral", "negative"];

export function ReviewPanel() {
  const [item, setItem] = useState<ReviewItem | null>(null);
  const [labels, setLabels] = useState<Labels | null>(null);
  const [stats, setStats] = useState<ReviewStats | null>(null);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    const [n, s] = await Promise.all([fetch("/api/review/next"), fetch("/api/review/stats")]);
    if (!n.ok) {
      setError(`Could not load the next filing (${n.status}).`);
      return;
    }
    const next = await n.json();
    if (s.ok) setStats((await s.json()) as ReviewStats);
    if (next.done) {
      setDone(true);
      setItem(null);
      return;
    }
    setItem(next as ReviewItem);
    setLabels((next as ReviewItem).proposed);
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  async function submit(action: "accept" | "correct" | "skip") {
    if (!item || !labels) return;
    setBusy(true);
    const r = await fetch("/api/review", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ accession: item.accession, action, label: action === "correct" ? labels : null }),
    });
    setBusy(false);
    if (!r.ok) {
      setError(`Saving failed (${r.status}).`);
      return;
    }
    await load();
  }

  const changed = item && labels ? (Object.keys(labels) as (keyof Labels)[]).some((k) => labels[k] !== item.proposed[k]) : false;

  return (
    <div className="grid">
      <div className="card">
        {error && <p className="error">{error}</p>}
        {done && <p>Every filing in the queue has been reviewed.</p>}
        {item && labels && (
          <>
            <div className="meta">
              {item.company} · {item.form} · <code>{item.accession}</code> · {item.remaining} left of {item.total}
            </div>
            <div style={{ maxHeight: 420, overflow: "auto", marginTop: 10 }}>
              {item.paragraphs.map((p, i) => (
                <p key={p.paragraph_id ?? i} className="para">
                  <span className="pid">[P{i + 1}] {sectionLabel(p.section)}</span>
                  {p.text}
                </p>
              ))}
            </div>
          </>
        )}
      </div>
      <div className="card">
        <h2>Labels</h2>
        {item && labels ? (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submit(changed ? "correct" : "accept");
            }}
          >
            <label className="field" htmlFor="guidance">
              Guidance direction
              <select id="guidance" value={labels.guidance_direction} onChange={(e) => setLabels({ ...labels, guidance_direction: e.target.value as Labels["guidance_direction"] })}>
                {GUIDANCE.map((g) => (
                  <option key={g} value={g}>
                    {g}
                  </option>
                ))}
              </select>
            </label>
            <label className="field" htmlFor="tone">
              Management tone
              <select id="tone" value={labels.management_tone} onChange={(e) => setLabels({ ...labels, management_tone: e.target.value as Labels["management_tone"] })}>
                {TONE.map((t) => (
                  <option key={t} value={t}>
                    {t}
                  </option>
                ))}
              </select>
            </label>
            <label className="field check" htmlFor="liquidity">
              <input id="liquidity" type="checkbox" checked={labels.liquidity_concern} onChange={(e) => setLabels({ ...labels, liquidity_concern: e.target.checked })} />
              Liquidity concern
            </label>
            <label className="field check" htmlFor="litigation">
              <input id="litigation" type="checkbox" checked={labels.litigation_risk} onChange={(e) => setLabels({ ...labels, litigation_risk: e.target.checked })} />
              Litigation risk
            </label>
            <div className="row" style={{ marginTop: 12 }}>
              <button className="primary" type="submit" disabled={busy}>
                {changed ? "Save correction" : "Accept"}
              </button>
              <button type="button" onClick={() => void submit("skip")} disabled={busy}>
                Skip
              </button>
            </div>
          </form>
        ) : (
          <p className="meta">Nothing to review.</p>
        )}
        {stats && (
          <>
            <h2 style={{ marginTop: 18 }}>Teacher agreement</h2>
            <p className="meta">
              {stats.reviewed} reviewed, {stats.corrected} corrected, {stats.skipped} skipped.
            </p>
            <table>
              <thead>
                <tr>
                  <th>Field</th>
                  <th className="num">Agreement</th>
                  <th className="num">Cohen&apos;s kappa</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(stats.fields).map(([f, v]) => (
                  <tr key={f}>
                    <td>{f.replaceAll("_", " ")}</td>
                    <td className="num">{v.agreement === null ? "n/a" : `${(v.agreement * 100).toFixed(0)}%`}</td>
                    <td className="num">{v.kappa === null || Number.isNaN(v.kappa) ? "n/a" : v.kappa.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      </div>
    </div>
  );
}
