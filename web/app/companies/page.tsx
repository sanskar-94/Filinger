import Link from "next/link";
import { backendJson } from "@/lib/backend";
import type { Company } from "@/lib/types";

export const dynamic = "force-dynamic";

export default async function Companies() {
  const companies = (await backendJson<Company[]>("/v1/companies")) ?? [];
  return (
    <div className="card">
      <h2>Companies ({companies.length})</h2>
      {companies.length === 0 ? (
        <p className="meta">The API is not reachable. Start it with flab serve and set FLAB_API_URL.</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Company</th>
              <th className="num">Filings</th>
              <th>Coverage</th>
            </tr>
          </thead>
          <tbody>
            {companies.map((c) => (
              <tr key={c.ticker}>
                <td>
                  <Link href={`/companies/${c.ticker}`}>{c.ticker}</Link>
                </td>
                <td>{c.company}</td>
                <td className="num">{c.filings}</td>
                <td className="meta">
                  {c.first} to {c.last}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
