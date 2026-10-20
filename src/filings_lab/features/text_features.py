"""Per section text features and year over year change features."""
from __future__ import annotations

import math
import re
from collections import Counter

from .lexicon import Lexicon, tokenize

_SENT = re.compile(r"(?<=[\.\!\?])\s+")
_NUM = re.compile(r"\$?\d[\d,]*\.?\d*%?")
_VOWELS = re.compile(r"[aeiouy]+")
GUIDANCE_UP = re.compile(r"\b(rais(e|es|ed|ing)|increas(e|es|ed|ing)|above the high end|boost(s|ed)?)\b[^.]{0,80}\b(guidance|outlook|forecast)\b"
                         r"|\b(guidance|outlook)\b[^.]{0,60}\b(rais(ed|ing)|increas(ed|ing))\b", re.I)
GUIDANCE_DOWN = re.compile(r"\b(lower(s|ed|ing)?|reduc(e|es|ed|ing)|cut(s|ting)?|below the low end)\b[^.]{0,80}\b(guidance|outlook|forecast)\b"
                           r"|\b(guidance|outlook)\b[^.]{0,60}\b(lowered|reduced|cut)\b", re.I)
GUIDANCE_FLAT = re.compile(r"\b(reaffirm(s|ed)?|maintain(s|ed)?|unchanged|reiterat(e|es|ed))\b[^.]{0,80}\b(guidance|outlook)\b"
                           r"|\b(guidance|outlook)\b[^.]{0,60}\b(unchanged|reaffirmed|maintained)\b", re.I)
GOING_CONCERN = re.compile(r"going concern|substantial doubt|covenant waiver|not in compliance with[^.]{0,40}covenant", re.I)

_LEX = Lexicon()


def syllables(word: str) -> int:
    word = word.lower().rstrip("e")
    return max(1, len(_VOWELS.findall(word)))


def section_features(text: str, lexicon: Lexicon | None = None) -> dict[str, float]:
    lex = lexicon or _LEX
    tokens = tokenize(text)
    n = len(tokens)
    sents = [s for s in _SENT.split(text) if s.strip()]
    complex_words = sum(1 for t in tokens if len(t) > 6 and syllables(t) >= 3)
    feats = {
        "n_words": float(n),
        "log_words": math.log1p(n),
        "avg_sent_len": n / max(len(sents), 1),
        "fog": 0.4 * (n / max(len(sents), 1) + 100 * complex_words / max(n, 1)),
        "num_density": len(_NUM.findall(text)) / max(n, 1),
        "fls_share": sum(1 for s in sents if any(w in s.lower() for w in (" expect", " anticipate", " outlook", " guidance", " will "))) / max(len(sents), 1),
        "guidance_up": float(bool(GUIDANCE_UP.search(text))),
        "guidance_down": float(bool(GUIDANCE_DOWN.search(text))),
        "guidance_flat": float(bool(GUIDANCE_FLAT.search(text))),
        "going_concern": float(bool(GOING_CONCERN.search(text))),
    }
    feats.update(lex.score(tokens))
    return feats


def _tf(text: str) -> Counter:
    return Counter(t for t in tokenize(text) if len(t) > 2)


def similarity(a: str, b: str) -> dict[str, float]:
    """Cosine (term frequency) and Jaccard similarity between two versions of a section."""
    ca, cb = _tf(a), _tf(b)
    dot = sum(v * cb.get(k, 0) for k, v in ca.items())
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    sa, sb = set(ca), set(cb)
    return {
        "sim_cosine": dot / (na * nb) if na and nb else float("nan"),
        "sim_jaccard": len(sa & sb) / len(sa | sb) if sa | sb else float("nan"),
        "len_change": math.log1p(sum(ca.values())) - math.log1p(sum(cb.values())),
    }
