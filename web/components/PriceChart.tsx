import type { PricePoint } from "@/lib/types";

/** Dependency free SVG line chart of weekly closes with filing reaction markers. */
export function PriceChart({ series, markers }: { series: PricePoint[]; markers: { date: string; up: boolean }[] }) {
  if (series.length < 2) return <p className="meta">No price history.</p>;
  const W = 760;
  const H = 220;
  const pad = 36;
  const ts = series.map((p) => new Date(p.date).getTime());
  const ys = series.map((p) => p.close);
  const x0 = Math.min(...ts);
  const x1 = Math.max(...ts);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  const sx = (t: number) => pad + ((t - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
  const sy = (v: number) => H - pad - ((v - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
  const path = series.map((p, i) => `${i ? "L" : "M"}${sx(ts[i] ?? x0).toFixed(1)},${sy(p.close).toFixed(1)}`).join(" ");
  const nearestClose = (t: number) => {
    let best = 0;
    for (let i = 1; i < ts.length; i++) if (Math.abs((ts[i] ?? 0) - t) < Math.abs((ts[best] ?? 0) - t)) best = i;
    return ys[best] ?? y0;
  };
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Weekly closing price with filing reactions">
      <line x1={pad} x2={W - pad} y1={H - pad} y2={H - pad} stroke="var(--border)" />
      <text x={pad} y={H - 12} fill="var(--muted)" fontSize="11">{series[0]?.date}</text>
      <text x={W - pad} y={H - 12} fill="var(--muted)" fontSize="11" textAnchor="end">{series.at(-1)?.date}</text>
      <text x={pad - 6} y={sy(y1) + 4} fill="var(--muted)" fontSize="11" textAnchor="end">{y1.toFixed(0)}</text>
      <text x={pad - 6} y={sy(y0) + 4} fill="var(--muted)" fontSize="11" textAnchor="end">{y0.toFixed(0)}</text>
      <path d={path} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
      {markers.map((m) => {
        const t = new Date(m.date).getTime();
        if (t < x0 || t > x1) return null;
        return <circle key={m.date + String(m.up)} cx={sx(t)} cy={sy(nearestClose(t))} r="3.2" fill={m.up ? "var(--pos)" : "var(--neg)"} />;
      })}
    </svg>
  );
}
