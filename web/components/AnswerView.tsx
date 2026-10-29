"use client";

import { parseAnswer } from "@/lib/citations";
import type { AgentAnswer } from "@/lib/types";

interface Props {
  answer: AgentAnswer;
  activeId: string | null;
  onCite: (id: string) => void;
}

export function AnswerView({ answer, activeId, onCite }: Props) {
  const segments = parseAnswer(answer.answer, answer.citations.map((c) => c.paragraph_id));
  return (
    <div className="answer" data-testid="answer">
      {segments.map((s, i) =>
        s.kind === "text" ? (
          <span key={i}>{s.value}</span>
        ) : (
          <button
            key={i}
            type="button"
            className={`cite${s.valid ? "" : " invalid"}${activeId === s.id ? " active" : ""}`}
            title={s.valid ? "Show the cited paragraph" : "This citation could not be verified against retrieved text"}
            onClick={() => s.valid && onCite(s.id)}
            aria-label={`citation ${s.n}`}
          >
            {s.n}
          </button>
        ),
      )}
    </div>
  );
}
