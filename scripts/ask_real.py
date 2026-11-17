"""Ask the analyst agent questions about real filings and write each answer beside the paragraphs it cites.

    python scripts/ask_real.py --lake data/real_pilot/lake --questions configs/real_questions.json \
        --out eval/real/agent_examples.md

The questions have no answer key, unlike the synthetic evaluation, so the output is built to be checked by
reading: every citation the agent makes is validated by the tool box (an id it was never shown is flagged), and
the full text of each cited paragraph, its filing and its date are printed under the answer. Uses the hosted
model of ``FLAB_OPENAI_MODEL`` on NVIDIA's free endpoints (a key in ``.env``) and BM25 retrieval, which needs no
embeddings of the real lake.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lake", default="data/real_pilot/lake")
    p.add_argument("--questions", default="configs/real_questions.json")
    p.add_argument("--out", default="eval/real/agent_examples.md")
    a = p.parse_args(argv)

    from filings_lab.agent.openai_agent import OpenAIChatAgent
    from filings_lab.agent.retrieval import BM25Index
    from filings_lab.agent.store import DataStore
    from filings_lab.agent.tools import ToolBox
    from filings_lab.config import load_dotenv
    from filings_lab.openai_compat import default_model

    load_dotenv()
    store = DataStore(a.lake)
    t0 = time.perf_counter()
    index = BM25Index(store.paragraphs)
    print(f"{len(store.paragraphs):,} paragraphs indexed in {time.perf_counter() - t0:.0f} s", flush=True)
    agent = OpenAIChatAgent(ToolBox(store, index))
    questions = json.loads(Path(a.questions).read_text())

    lines = [f"# The analyst agent on real filings ({default_model()}, BM25)", "",
             f"{len(questions)} questions about real companies, asked of the lake built from their EDGAR filings "
             f"(`{a.lake}`). There is no answer key: each answer is followed by the full text of every paragraph it "
             "cites, so it can be judged against its sources. Market reactions come from the event table "
             "(the 3 day abnormal return against SPY).", ""]
    for n, q in enumerate(questions, start=1):
        ans = agent.ask(q["question"], q.get("ticker"))
        print(f"{n}/{len(questions)} {q['ticker']}: {len(ans.citations)} citations, {len(ans.invalid_citations)} invalid, "
              f"{ans.turns} turns{', error ' + ans.error if ans.error else ''}", flush=True)
        lines += [f"## {n}. {q['question']}", "", f"Company in focus: {q['ticker']}. Turns: {ans.turns}. "
                  f"Tool calls: {', '.join(c['tool'] for c in ans.tool_calls) or 'none'}.", ""]
        lines += [(ans.answer or f"(no answer: {ans.error})").strip(), ""]
        if ans.invalid_citations:
            lines += [f"Citations the agent was never shown, flagged: {', '.join(ans.invalid_citations)}", ""]
        for c in ans.citations:
            row = store.by_id.loc[c.paragraph_id]
            date = str(row["date_filed"])[:10]
            lines += [f"> **[cite:{c.paragraph_id}]** {row['ticker']} {row['form']} filed {date}, section "
                      f"`{row['section']}`, accession {row['accession']}", ">",
                      "> " + str(row["text"]).replace("\n", " "), ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    print(f"-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
