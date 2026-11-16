"""``flab embed``, ``flab retrieval-eval`` and ``flab agent-eval``."""
from __future__ import annotations

import json


def _embed(a):
    from .embeddings import cache_path, embed_texts, get_embedder
    from .store import DataStore

    paragraphs = DataStore(a.lake).paragraphs
    embedder = get_embedder(a.model)
    path = cache_path(a.lake, embedder.name)
    _, _, stats = embed_texts(paragraphs["text"].fillna("").tolist(), embedder, path, log=print)
    print(json.dumps({**stats, "cache": str(path)}))


def _register_embed(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--model", default=None, help="embedding model (default BAAI/bge-small-en-v1.5; hash-512 needs no download)")
    p.set_defaults(fn=_embed)


def _retrieval_eval(a):
    from .embeddings import get_embedder
    from .retrieval_eval import evaluate
    from .store import DataStore

    modes = [m.strip() for m in a.systems.split(",") if m.strip()]
    embedder = get_embedder(a.model) if any(m != "bm25" for m in modes) else None
    summary = evaluate(DataStore(a.lake).paragraphs, a.lake, a.out, modes, a.companies_per_need, a.seed, embedder)
    print(open(f"{a.out}/report.md").read())
    print(json.dumps(summary["info"]))


def _register_retrieval_eval(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--out", default="eval/retrieval")
    p.add_argument("--systems", default="bm25,dense,hybrid")
    p.add_argument("--model", default=None)
    p.add_argument("--companies-per-need", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_retrieval_eval)


def _agent_eval(a):
    from .agent_eval import evaluate

    summary = evaluate(a.lake, a.truth, a.out, modes=[m.strip() for m in a.retrieval.split(",") if m.strip()],
                       agent=a.agent, per_type=a.per_type, seed=a.seed, embed_model=a.model, workers=a.workers)
    print(open(f"{a.out}/report.md").read())
    print(json.dumps(summary["info"]))


def _register_agent_eval(p):
    p.add_argument("--lake", default="data/lake")
    p.add_argument("--truth", default="data/synthetic/truth.parquet")
    p.add_argument("--out", default="eval/agent")
    p.add_argument("--agent", choices=["offline", "claude", "openai"], default="offline",
                   help="offline: the extractive agent; claude: the hosted model through the tool loop; openai: any OpenAI "
                        "compatible API, NVIDIA's free endpoints by default (both hosted modes need a key)")
    p.add_argument("--retrieval", default="bm25,hybrid", help="retrieval modes to compare, comma separated")
    p.add_argument("--model", default=None, help="embedding model for dense retrieval")
    p.add_argument("--per-type", type=int, default=40, help="questions per question type")
    p.add_argument("--workers", type=int, default=1,
                   help="questions asked at once with --agent openai; the client still keeps to the API's request rate")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=_agent_eval)


def _api_models(a):
    from ..openai_compat import ChatClient, default_model

    for model in ChatClient().models():
        print(model + ("   (default)" if model == default_model() else ""))


COMMANDS = {"api-models": (lambda p: p.set_defaults(fn=_api_models), "list the models the OpenAI compatible API serves right now"),
            "embed": (_register_embed, "embed the lake's paragraphs into the dense retrieval cache"),
            "retrieval-eval": (_register_retrieval_eval, "compare BM25, dense and hybrid retrieval on the synthetic world"),
            "agent-eval": (_register_agent_eval, "score the analyst agent on questions with known answers")}
