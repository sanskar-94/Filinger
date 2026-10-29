"use client";

import { useEffect, useState } from "react";
import type { Citation, Paragraph } from "@/lib/types";

export function SourcePanel({ citations, activeId }: { citations: Citation[]; activeId: string | null }) {
  const [para, setPara] = useState<Paragraph | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!activeId) return;
    let cancelled = false;
    fetch(`/api/paragraphs/${activeId}`)
      .then(async (r) => {
        if (!r.ok) throw new Error(`paragraph lookup failed (${r.status})`);
        return (await r.json()) as Paragraph;
      })
      .then((p) => {
        if (!cancelled) {
          setPara(p);
          setError(null);
        }
      })
      .catch((e: Error) => {
        if (!cancelled) setError(e.message);
      });
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  return (
    <div className="card">
      <h2>Sources</h2>
      {activeId && para && para.paragraph_id === activeId ? (
        <div>
          <div className="meta">
            {para.company} ({para.ticker}) · {para.form} filed {para.date_filed} · {para.section}
          </div>
          <div className="quote">{para.text}</div>
          <a href={para.url} target="_blank" rel="noreferrer">
            Open the filing on EDGAR
          </a>
        </div>
      ) : error ? (
        <p className="error">{error}</p>
      ) : (
        <p className="meta">Click a numbered citation to read the exact paragraph it comes from.</p>
      )}
      {citations.length > 0 && (
        <ol className="sources">
          {citations.map((c) => (
            <li key={c.paragraph_id} className="meta">
              {c.form} {c.date_filed} · {c.section} ·{" "}
              <a href={c.url} target="_blank" rel="noreferrer">
                {c.accession}
              </a>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
