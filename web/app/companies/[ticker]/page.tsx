import { PriceChart } from "@/components/PriceChart";
import { backendJson } from "@/lib/backend";
import { formatPct } from "@/lib/citations";
import type { FilingRow, PricePoint } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function CompanyPage({ params }: { params: Promise<{ ticker: string }> }) {
  const { ticker } = await params;
  const t = ticker.toUpperCase();
  const [filings, prices] = await Promise.all([
    backendJson<{ filings: FilingRow[] }>(`/v1/companies/${t}/filings?limit=200`),
    backendJson<{ series: PricePoint[] }>(`/v1/companies/${t}/prices`),
  ]);
  const rows = filings?.filings ?? [];
  const markers = rows.filter((f) => f.car_3d !== null).map((f) => ({ date: f.date_filed, up: (f.car_3d ?? 0) > 0 }));
  return (
    <>
      <h1 style={{ fontSize: 22 }}>{t}</h1>
      <div className="card">
        <h2>Weekly close with filing reactions (green: positive 3 day abnormal return, red: negative)</h2>
        <PriceChart series={prices?.series ?? []} markers={markers} />
      </div>
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Filings</h2>
        <table>
          <thead>
            <tr>
              <th>Filed</th>
              <th>Form</th>
              <th>8-K items</th>
              <th>Accession</th>
              <th className="num">3 day CAR</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((f) => (
              <tr key={f.accession}>
                <td>{f.date_filed}</td>
                <td>{f.form}</td>
                <td className="meta">{f.items ?? ""}</td>
                <td>
                  <code>{f.accession}</code>
                </td>
                <td className={`num ${f.car_3d === null ? "" : f.car_3d > 0 ? "pos" : "neg"}`}>{formatPct(f.car_3d)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
