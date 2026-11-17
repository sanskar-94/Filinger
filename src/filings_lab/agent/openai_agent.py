"""The analyst agent on any OpenAI compatible chat API (see ``filings_lab/openai_compat.py``).

The same loop as ``ClaudeAgent``: the model calls the lab's tools, every paragraph id a tool returned is
recorded, a citation of an id it was never shown gets one repair turn, and whatever is still invalid is
returned flagged. What differs is the wire format (``tool_calls`` on the assistant message, one ``tool``
message per result) and what an open model needs forgiven: arguments left out or invented, arguments that are
not valid JSON, the text "None" for an empty argument, reasoning printed inside ``<think>`` tags, and
citations written with brackets around the id or several to a bracket, and a reply with nothing in it. On the last turn the tools are withdrawn and the model is told
to answer from what it has, so a search that never ends still produces an answer.
"""
from __future__ import annotations

import json
import re
import time

from ..openai_compat import ChatClient, ChatError, default_model
from .agent import OUT_OF_TURNS, SYSTEM_PROMPT, AgentAnswer, build_citations
from .tools import TOOL_DEFS, ToolBox

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
# a citation as open models have written it: [cite:<id>], [cite: "id"], 【cite:id】, [cite:id; cite:id]
_CITE_GROUP = re.compile(r"[\[\u3010\u3014]\s*cite\s*:([^\]\u3011\u3015]*)[\]\u3011\u3015]", re.I)
_PARAGRAPH_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{16}(?![0-9a-f])", re.I)
EMPTY_REPLY = "Your last reply was empty. Answer the question now from what the tools returned, with citations."
CITATION_FORMAT = ("\n\nWrite each citation exactly like [cite:3fa2b1c4d5e6f708]: square brackets, the word cite, a colon and the "
                   "16 character paragraph id, with nothing around the id.")


def tidy_answer(text: str) -> str:
    """The model's final text without printed reasoning and with citations in the one form the validator reads."""
    def one_form(m: re.Match) -> str:
        ids = _PARAGRAPH_ID.findall(m.group(1))
        return "".join(f"[cite:{i.lower()}]" for i in ids) if ids else m.group(0)

    return _CITE_GROUP.sub(one_form, _THINK.sub("", text or "")).strip()


def openai_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in TOOL_DEFS]


_EMPTY = {"none", "null", "nil", "n/a", ""}


def clean_arguments(name: str, raw) -> dict:
    """A tool call's arguments as the tool accepts them.

    Parsed, without keys the tool does not declare, and with two habits of open models undone: the text "None"
    or "null" where a nullable argument should be empty (seen on the first live calls, where it turned every
    optional filter into a filter on the word None), and a number sent as text.
    """
    args = json.loads(raw or "{}") if isinstance(raw, str) else dict(raw or {})
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    declared = next((t["input_schema"]["properties"] for t in TOOL_DEFS if t["name"] == name), None)
    if declared is None:
        return args
    out = {}
    for key, value in args.items():
        if key not in declared:
            continue
        kind = declared[key].get("type")
        if isinstance(value, str):
            if isinstance(kind, list) and "null" in kind and value.strip().lower() in _EMPTY:
                value = None
            elif kind == "integer" and value.strip().lstrip("-").isdigit():
                value = int(value)
        out[key] = value
    return out


class OpenAIChatAgent:
    def __init__(self, toolbox: ToolBox, model: str | None = None, client: ChatClient | None = None, max_turns: int = 8,
                 max_tokens: int = 4096):
        self.tb, self.model, self.max_turns, self.max_tokens = toolbox, model or default_model(), max_turns, max_tokens
        self.client = client or ChatClient()
        self.tools = openai_tools()

    def ask(self, question: str, ticker: str | None = None) -> AgentAnswer:
        t0 = time.perf_counter()
        self.tb.reset()
        content = question if not ticker else f"{question}\n\n(Company in focus: ticker {ticker.upper()})"
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT + CITATION_FORMAT}, {"role": "user", "content": content}]
        calls: list[dict] = []
        tokens_in = tokens_out = turns = 0
        repaired, nudged, final_text, stop = False, False, "", None
        mode = f"openai:{self.model}"
        try:
            while turns < self.max_turns:
                turns += 1
                last = turns == self.max_turns
                if last and messages[-1]["role"] == "tool":
                    # a model that searches until the budget runs out must still conclude, most of all when the
                    # honest answer is that the filings do not say; without this the last turn is one more search
                    messages.append({"role": "user", "content": OUT_OF_TURNS})
                data = self.client.complete(model=self.model, messages=messages, tools=self.tools,
                                            tool_choice="none" if last else "auto", temperature=0.0, max_tokens=self.max_tokens)
                usage = data.get("usage") or {}
                tokens_in += usage.get("prompt_tokens") or 0
                tokens_out += usage.get("completion_tokens") or 0
                choice = data["choices"][0]
                msg, stop = choice["message"], choice.get("finish_reason")
                tool_calls = [] if last else (msg.get("tool_calls") or [])
                if tool_calls:
                    messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
                    for n, call in enumerate(tool_calls):
                        fn = call.get("function") or {}
                        name, c0, err = fn.get("name", ""), time.perf_counter(), None
                        try:
                            args = clean_arguments(name, fn.get("arguments"))
                            out = self.tb.execute(name, args)
                        except Exception as exc:  # noqa: BLE001 - report to the model, let it recover
                            args, out, err = {}, f"Error: {exc}", str(exc)
                        calls.append({"tool": name, "input": args, "ms": round((time.perf_counter() - c0) * 1000, 1), "error": err})
                        messages.append({"role": "tool", "tool_call_id": call.get("id") or f"call_{turns}_{n}", "content": out})
                    continue
                final_text = tidy_answer(msg.get("content") or "")
                if not final_text and not nudged and not last:
                    # seen live: no tool call and no text, from a model that spent the turn reasoning
                    nudged = True
                    messages.append({"role": "user", "content": EMPTY_REPLY})
                    continue
                _, invalid = build_citations(final_text, self.tb)
                if invalid and not repaired and turns < self.max_turns:
                    repaired = True
                    messages.append({"role": "assistant", "content": final_text})
                    messages.append({"role": "user", "content": (
                        "These citations were not returned by any tool in this conversation: "
                        + ", ".join(invalid) + ". Rewrite the answer citing only retrieved paragraphs; search again if needed.")})
                    continue
                break
        except (ChatError, KeyError, IndexError, OSError) as exc:
            return AgentAnswer("", [], [], calls, mode, turns, time.perf_counter() - t0, error=f"api: {exc}")
        valid, invalid = build_citations(final_text, self.tb)
        return AgentAnswer(final_text, valid, invalid, calls, mode, turns, time.perf_counter() - t0, tokens_in, tokens_out,
                           0.0, stop, meta={"repaired": repaired})
