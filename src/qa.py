from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


from router import route_query, decide_query_scope

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

INDEX_DIR = ROOT / "index" / "faiss_pdf"
LOG_PATH = ROOT / "logs" / "qa_log.jsonl"

DEBUG_RETRIEVAL = True


@dataclass
class SourceRef:
    n: int
    chunk_id: str
    doc_id: str
    page: int
    method: str
    preview: str


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _write_log(record: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_vectorstore() -> FAISS:
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"FAISS index not found at {INDEX_DIR}. Build it first.")

    embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(
        str(INDEX_DIR),
        embedder,
        allow_dangerous_deserialization=True,
    )
    return db


def _all_docs_from_docstore(db: FAISS) -> List[Document]:
    # internal, but standard in LC FAISS
    return list(db.docstore._dict.values())


def _tokenize(text: str) -> List[str]:
    # BM25 tokenizer
    if not text:
        return []
    return re.findall(r"[0-9A-Za-zÁ-ž&._-]{2,}", text.lower())


def _doc_key(d: Document) -> Tuple[str, str, int]:
    md = d.metadata or {}
    return (
        str(md.get("chunk_id", "")),
        str(md.get("doc_id", "")),
        int(md.get("page", 0) or 0),
    )


def _rrf_fuse(rankings: List[List[Document]], k: int = 60) -> List[Document]:
    """
    Reciprocal Rank Fusion:
    score(doc) += 1 / (k + rank)
    """
    scores: Dict[Tuple[str, str, int], float] = {}
    keep: Dict[Tuple[str, str, int], Document] = {}

    for docs in rankings:
        for rank, d in enumerate(docs, start=1):
            key = _doc_key(d)
            keep[key] = d
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [keep[key] for key, _ in fused]


def _build_context(docs: List[Document]) -> Tuple[str, List[SourceRef]]:
    sources: List[SourceRef] = []
    blocks: List[str] = []

    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        chunk_id = str(md.get("chunk_id", ""))
        doc_id = str(md.get("doc_id", ""))
        page = int(md.get("page", 0) or 0)
        method = str(md.get("method", ""))

        text = (d.page_content or "").strip()
        preview = text[:180].replace("\n", " ")

        sources.append(
            SourceRef(
                n=i,
                chunk_id=chunk_id,
                doc_id=doc_id,
                page=page,
                method=method,
                preview=preview,
            )
        )

        # numbered chunks for the LLM
        blocks.append(f"[{i}] {text}")

    return "\n\n".join(blocks), sources


def _normalize_citations(answer: str, sources: List[SourceRef]) -> Tuple[str, List[SourceRef]]:
    """
    Keep only sources that are actually cited in the answer.
    - Renumber citations to be continuous from [1]
    - Drop citations that don't exist in sources
    - Return (new_answer, new_sources)
    """
    if not answer:
        return answer, []

    nums = re.findall(r"\[(\d+)\]", answer)
    used: List[int] = []
    for n in nums:
        try:
            used.append(int(n))
        except Exception:
            continue

    # keep only valid
    used = [n for n in used if 1 <= n <= len(sources)]
    if not used:
        return answer.strip(), []

    # unique while preserving order
    used_unique: List[int] = []
    seen = set()
    for n in used:
        if n not in seen:
            seen.add(n)
            used_unique.append(n)

    mapping = {old: new for new, old in enumerate(used_unique, start=1)}

    def repl(m: re.Match) -> str:
        old = int(m.group(1))
        if old in mapping:
            return f"[{mapping[old]}]"
        # citation to non-existing source -> remove
        return ""

    new_answer = re.sub(r"\[(\d+)\]", repl, answer).strip()

    new_sources: List[SourceRef] = []
    for old in used_unique:
        s = sources[old - 1]
        new_sources.append(
            SourceRef(
                n=mapping[old],
                chunk_id=s.chunk_id,
                doc_id=s.doc_id,
                page=s.page,
                method=s.method,
                preview=s.preview,
            )
        )

    return new_answer, new_sources


def answer_question(
    question: str,
    top_k: int = 6,
    pool_k: int = 30,
    router_max_docs_single: int = 2,
    router_max_docs_multi: int = 6,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    db = load_vectorstore()
    all_docs = _all_docs_from_docstore(db)

    # 1) Decide scope: single vs multi
    scope_route = decide_query_scope(question, llm_model=model)

    # 2) Route to doc_ids (single vs multi max)
    max_docs = router_max_docs_multi if scope_route.scope == "multi" else router_max_docs_single

    route = route_query(
        question,
        single_max_docs=router_max_docs_single,
        multi_max_docs=router_max_docs_multi,
        prefer_llm=True,
        llm_model=model,
    )
    routed_doc_ids = route.doc_ids[:max_docs]

    # --- 3) HARD SCOPE retrieval universe (critical fix) ---
    # If router found doc_ids and they exist in the index -> restrict retrieval universe to them.
    used_routing = False
    universe_docs = all_docs

    if routed_doc_ids:
        scoped_universe = [d for d in all_docs if (d.metadata or {}).get("doc_id") in routed_doc_ids]
        if scoped_universe:
            universe_docs = scoped_universe
            used_routing = True

    # --- Retrieval pool: BM25 + Vector ---
    bm25 = BM25Retriever.from_documents(universe_docs, preprocess_func=_tokenize)
    bm25.k = pool_k
    bm25_docs = bm25.invoke(question)

    # vector search is over whole FAISS index; get a bigger pool then filter if scoped
    vec_docs = db.similarity_search(question, k=pool_k * 5)
    if used_routing:
        vec_docs = [d for d in vec_docs if (d.metadata or {}).get("doc_id") in routed_doc_ids]

    pool_docs = _rrf_fuse([bm25_docs, vec_docs], k=60)
    docs = pool_docs[:top_k]

    if DEBUG_RETRIEVAL:
        print("TOP DOCS (sent to LLM):")
        for d in docs:
            md = d.metadata or {}
            print("-", md.get("doc_id"), "p", md.get("page"), md.get("chunk_id"), "|", md.get("method"))

        print("QUERY_SCOPE =", scope_route.scope)
        print("SCOPE_CONFIDENCE =", scope_route.confidence)
        print("SCOPE_REASON =", scope_route.reason)

        print("ROUTED_DOC_IDS =", routed_doc_ids)
        print("USED_ROUTING =", used_routing)
        print("ROUTE_USED_LLM =", route.used_llm)
        print("ROUTE_CONFIDENCE =", route.confidence)
        print("ROUTE_REASON =", route.reason)

    context, sources = _build_context(docs)

    system = (
        "You are a careful assistant answering questions ONLY from the provided context.\n"
        "Rules:\n"
        "- Use ONLY the context. If the answer is not in the context, say you don't know.\n"
        "- Do not invent facts.\n"
        "- Citations MUST use bracket numbers like [1], [2] that correspond to the context chunks.\n"
        "- IMPORTANT: Every factual sentence MUST end with at least one citation.\n"
        "- If a sentence would require citations from different chunks, split it into multiple sentences/bullets so each ends with its own citation.\n"
        "- Do NOT put citations only at the end of a paragraph.\n"
        "- Do NOT cite file names in-text; only use bracket citations.\n"
        "- Prefer short bullet points; each bullet should be one claim + citations.\n"
        "- Answer in Czech/Slovak (match the question language)."
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"CONTEXT CHUNKS:\n{context}\n\n"
        "Write a helpful answer. Include citations."
    )

    llm = ChatOpenAI(model=model, temperature=temperature)
    resp = llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    answer = (getattr(resp, "content", "") or "").strip()

    # 4) Keep only used citations/sources, renumber from [1]
    answer, sources = _normalize_citations(answer, sources)

    _write_log(
        {
            "ts": _now_iso(),
            "question": question,
            "top_k": top_k,
            "pool_k": pool_k,
            "model": model,
            "temperature": temperature,
            "answer": answer,
            "sources": [asdict(s) for s in sources],
            "scope_routing": asdict(scope_route),
            "routing": {
                "doc_ids": routed_doc_ids,
                "used_routing": used_routing,
                "router_used_llm": route.used_llm,
                "router_confidence": route.confidence,
                "router_reason": route.reason,
            },
            "retrieval": "hard_scope_universe + rrf(bm25+faiss) + citation_normalize",
        }
    )

    return {
        "answer": answer,
        "sources": [asdict(s) for s in sources],
        "used_top_k": len(docs),
        "query_scope": scope_route.scope,
        "scope_confidence": scope_route.confidence,
        "routing_doc_ids": routed_doc_ids,
        "used_routing": used_routing,
        "router_used_llm": route.used_llm,
        "router_confidence": route.confidence,
    }


if __name__ == "__main__":
    q = input("Zadejte otázku: ").strip()
    out = answer_question(
        q,
        top_k=6,
        pool_k=30,
        router_max_docs_single=2,
        router_max_docs_multi=6,
        model="gpt-4o-mini",
        temperature=0.0,
    )

    print("\n" + out["answer"] + "\n")

    print("Zdroje:")
    for s in out["sources"]:
        print(f"[{s['n']}] {s['doc_id']} (strana {s['page']}), chunk {s['chunk_id']}, method {s['method']}")
