import { backendJson } from "@/lib/backend";
import { axisPct, formatPct } from "@/lib/citations";
import type { Readout } from "@/lib/types";

export const dynamic = "force-dynamic";

const pts = (v: number) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(2)} pts`;

function Interval({ label, lo, hi, est, axis }: { label: string; lo: number; hi: number; est: number; axis: [number, number] }) {
  const [a, b] = axis;
  return (
    <div className="ivl">
      <div className="ivl-label">{label}</div>
      <div className="ivl-track" role="img" aria-label={`${label}: ${pts(est)} from ${pts(lo)} to ${pts(hi)}`}>
        <span className="ivl-zero" style={{ left: `${axisPct(0, a, b)}%` }} />
        <span className={`ivl-bar${lo > 0 ? " pos" : hi < 0 ? " neg" : ""}`} style={{ left: `${axisPct(lo, a, b)}%`, width: `${axisPct(hi, a, b) - axisPct(lo, a, b)}%` }} />
        <span className="ivl-dot" style={{ left: `${axisPct(est, a, b)}%` }} />
      </div>
      <div className="ivl-text">
        {pts(est)} [{pts(lo)}, {pts(hi)}]
      </div>
    </div>
  );
}

export default async function ExperimentPage() {
  const r = await backendJson<Readout>("/v1/experiments/current/readout");
  if (!r) return <p className="meta">No experiment readout yet: the API is unreachable or too few answers have been rated.</p>;
  const p = r.primary;
  const arms = [r.control, r.treatment];
  const lo = Math.min(p.ci_clustered[0], p.msprt_confidence_sequence[0], 0);
  const hi = Math.max(p.ci_clustered[1], p.msprt_confidence_sequence[1], 0);
  const pad = (hi - lo) * 0.1;
  const axis: [number, number] = [lo - pad, hi + pad];
  const tone = r.decision === "ship treatment" ? "pos" : r.decision === "do not ship" ? "neg" : "";
  return (
    <>
      <h1 style={{ fontSize: 22, margin: "0 0 4px" }}>Experiment: {r.experiment}</h1>
      <p className="meta" style={{ marginTop: 0 }}>
        Live readout of the A/B test running in this app ({p.mode}). Analysts are randomised by browser; the decision uses intervals clustered by analyst.
      </p>
      <div className="card">
        <div className="row">
          <span className={`badge ${tone}`}>decision: {r.decision}</span>
          <span className="meta">{r.reasons.join(" ")}</span>
        </div>
        <div className="row" style={{ marginTop: 14, gap: 28 }}>
          <div>
            <div className="meta">Lift in thumbs up rate, {r.treatment} minus {r.control}</div>
            <div className="big">{pts(p.diff)}</div>
          </div>
          <div>
            <div className="meta">Always valid p value</div>
            <div className="big">{p.msprt_p_final.toFixed(3)}</div>
          </div>
          <div>
            <div className="meta">Relative lift</div>
            <div className="big">{formatPct(p.relative_lift, 1)}</div>
          </div>
        </div>
        <div style={{ marginTop: 14 }}>
          <Interval label="Clustered 95% interval" lo={p.ci_clustered[0]} hi={p.ci_clustered[1]} est={p.diff} axis={axis} />
          <Interval label="Always valid interval" lo={p.msprt_confidence_sequence[0]} hi={p.msprt_confidence_sequence[1]} est={p.diff} axis={axis} />
        </div>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Arms</h2>
        <table>
          <thead>
            <tr>
              <th>Variant</th>
              <th>What it serves</th>
              <th className="num">Analysts</th>
              <th className="num">Rated</th>
              <th className="num">Thumbs up rate</th>
              <th className="num">Wilson 95%</th>
              <th className="num">p95 latency</th>
            </tr>
          </thead>
          <tbody>
            {arms.map((v) => {
              const a = r.arms[v];
              if (!a) return null;
              return (
                <tr key={v}>
                  <td>{v}</td>
                  <td className="meta">{r.descriptions[v] ?? ""}</td>
                  <td className="num">{a.users.toLocaleString()}</td>
                  <td className="num">{a.rated.toLocaleString()}</td>
                  <td className="num">{(a.thumbs_up_rate * 100).toFixed(1)}%</td>
                  <td className="num">
                    {(a.wilson_low * 100).toFixed(1)}% to {(a.wilson_high * 100).toFixed(1)}%
                  </td>
                  <td className="num">{(a.latency_p95_s * 1000).toFixed(0)} ms</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Checks</h2>
        <ol className="sources">
          <li className="meta">
            Traffic split: {Object.entries(r.srm.users).map(([k, n]) => `${k} ${n.toLocaleString()}`).join(", ")} analysts, chi square p = {r.srm.p_value.toFixed(3)}{" "}
            {r.srm.mismatch ? "(mismatch)" : "(ok)"}
          </li>
          {Object.entries(r.guardrails).map(([k, g]) => (
            <li key={k} className="meta">
              {k.replaceAll("_", " ")}: {formatPct(g.estimate)} against a limit of {formatPct(g.limit, 0)}, {g.breached ? "breached" : "ok"}
            </li>
          ))}
          <li className="meta">Design effect from repeat raters: {p.design_effect.toFixed(2)}</li>
        </ol>
      </div>
    </>
  );
}
