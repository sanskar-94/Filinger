"""The analyst agent on any OpenAI compatible chat API (see ``filings_lab/openai_compat.py``).

The same loop as ``ClaudeAgent``: the model calls the lab's tools, every paragraph id a tool returned is
recorded, a citation of an id it was never shown gets one repair turn, and whatever is still invalid is
returned flagged. What differs is the wire format (``tool_calls`` on the assistant message, one ``tool``
message per result) and what an open model needs forgiven: arguments left out or invented, arguments that are
not valid JSON, and reasoning printed inside ``<think>`` tags.
"""
from __future__ import annotations

import json
import re
import time

from ..openai_compat import ChatClient, ChatError, default_model
from .agent import SYSTEM_PROMPT, AgentAnswer, build_citations
from .tools import TOOL_DEFS, ToolBox

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def openai_tools() -> list[dict]:
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
            for t in TOOL_DEFS]


def clean_arguments(name: str, raw) -> dict:
    """A tool call's arguments as the tool accepts them: parsed, and without keys the tool does not declare."""
    args = json.loads(raw or "{}") if isinstance(raw, str) else dict(raw or {})
    if not isinstance(args, dict):
        raise ValueError("arguments must be a JSON object")
    declared = next((t["input_schema"]["properties"] for t in TOOL_DEFS if t["name"] == name), None)
    return args if declared is None else {k: v for k, v in args.items() if k in declared}


class OpenAIChatAgent:
    def __init__(self, toolbox: ToolBox, model: str | None = None, client: ChatClient | None = None, max_turns: int = 8,
                 max_tokens: int = 2000):
        self.tb, self.model, self.max_turns, self.max_tokens = toolbox, model or default_model(), max_turns, max_tokens
        self.client = client or ChatClient()
        self.tools = openai_tools()

    def ask(self, question: str, ticker: str | None = None) -> AgentAnswer:
        t0 = time.perf_counter()
        self.tb.reset()
        content = question if not ticker else f"{question}\n\n(Company in focus: ticker {ticker.upper()})"
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]
        calls: list[dict] = []
        tokens_in = tokens_out = turns = 0
        repaired, final_text, stop = False, "", None
        mode = f"openai:{self.model}"
        try:
            while turns < self.max_turns:
                turns += 1
                data = self.client.complete(model=self.model, messages=messages, tools=self.tools, tool_choice="auto",
                                            temperature=0.0, max_tokens=self.max_tokens)
                usage = data.get("usage") or {}
                tokens_in += usage.get("prompt_tokens") or 0
                tokens_out += usage.get("completion_tokens") or 0
                choice = data["choices"][0]
                msg, stop = choice["message"], choice.get("finish_reason")
                tool_calls = msg.get("tool_calls") or []
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
                final_text = _THINK.sub("", msg.get("content") or "").strip()
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
