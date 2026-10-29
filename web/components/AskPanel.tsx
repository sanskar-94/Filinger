"use client";

import { useEffect, useState } from "react";
import { AnswerView } from "@/components/AnswerView";
import { SourcePanel } from "@/components/SourcePanel";
import type { AgentAnswer, Company } from "@/lib/types";

const EXAMPLES = [
  "Did the company change its revenue guidance, and how did the stock react?",
  "Has management flagged any liquidity or going concern issues?",
  "Summarise any litigation or regulatory investigations disclosed in recent filings.",
  "How has management described demand and margins in recent quarters?",
];

export function AskPanel() {
  const [companies, setCompanies] = useState<Company[]>([]);
  const [ticker, setTicker] = useState("");
  const [question, setQuestion] = useState(EXAMPLES[0] ?? "");
  const [answer, setAnswer] = useState<AgentAnswer | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rated, setRated] = useState<number | null>(null);

  useEffect(() => {
    fetch("/api/companies")
      .then((r) => (r.ok ? r.json() : []))
      .then((c: Company[]) => {
        setCompanies(c);
        if (c[0]) setTicker(c[0].ticker);
      })
      .catch(() => setCompanies([]));
  }, []);

  async function ask() {
    setLoading(true);
    setError(null);
    setAnswer(null);
    setActiveId(null);
    setRated(null);
    try {
      const r = await fetch("/api/ask", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question, ticker: ticker || null }),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail ?? data.error ?? `request failed (${r.status})`);
      const a = data as AgentAnswer;
      setAnswer(a);
      setActiveId(a.citations[0]?.paragraph_id ?? null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  async function rate(rating: 1 | -1) {
    if (!answer) return;
    const r = await fetch("/api/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ request_id: answer.request_id, rating }),
    });
    if (r.ok) setRated(rating);
  }

  return (
    <div className="grid">
      <div>
        <div className="card">
          <div className="row">
            <label htmlFor="ticker" className="meta">
              Company
            </label>
            <select id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value)}>
              <option value="">All companies</option>
              {companies.map((c) => (
                <option key={c.ticker} value={c.ticker}>
                  {c.ticker} · {c.company}
                </option>
              ))}
            </select>
          </div>
          <textarea
            aria-label="Question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void ask();
            }}
            style={{ marginTop: 10 }}
          />
          <div className="row" style={{ marginTop: 10, justifyContent: "space-between" }}>
            <div className="chips">
              {EXAMPLES.slice(1).map((q) => (
                <button key={q} type="button" className="chip" onClick={() => setQuestion(q)}>
                  {q.length > 48 ? `${q.slice(0, 48)}...` : q}
                </button>
              ))}
            </div>
            <button className="primary" type="button" onClick={() => void ask()} disabled={loading || question.length < 3}>
              {loading ? "Researching..." : "Ask"}
            </button>
          </div>
        </div>

        {error && <p className="error">{error}</p>}

        {answer && (
          <div className="card" style={{ marginTop: 16 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h2 style={{ margin: 0 }}>Answer</h2>
              <div className="row">
                <span className="badge">{answer.mode}</span>
                <span className="badge">variant {answer.variant}</span>
                <span className="badge">{answer.latency_s.toFixed(2)} s</span>
                {answer.cost_usd > 0 && <span className="badge">${answer.cost_usd.toFixed(4)}</span>}
              </div>
            </div>
            <AnswerView answer={answer} activeId={activeId} onCite={setActiveId} />
            {answer.invalid_citations.length > 0 && (
              <p className="error">
                {answer.invalid_citations.length} citation(s) could not be verified and are marked with a dashed outline.
              </p>
            )}
            <div className="row" style={{ marginTop: 12 }}>
              <span className="meta">Was this helpful?</span>
              <button type="button" onClick={() => void rate(1)} disabled={rated !== null} aria-label="helpful">
                Yes
              </button>
              <button type="button" onClick={() => void rate(-1)} disabled={rated !== null} aria-label="not helpful">
                No
              </button>
              {rated !== null && <span className="meta">Thanks, recorded.</span>}
            </div>
            <details style={{ marginTop: 12 }}>
              <summary>
                {answer.tool_calls.length} tool call(s), {answer.turns} turn(s)
              </summary>
              <pre>{JSON.stringify(answer.tool_calls, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>
      <SourcePanel citations={answer?.citations ?? []} activeId={activeId} />
    </div>
  );
}
