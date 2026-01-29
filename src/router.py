from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CONTRACTS_PATH = ROOT / "data" / "processed" / "contracts.json"


# Data models

@dataclass
class RouteResult:
    doc_ids: List[str]
    used_llm: bool
    confidence: float
    reason: str
    matched_signals: Dict[str, Any]


@dataclass
class ScopeResult:
    scope: str  # "single" | "multi"
    used_llm: bool
    confidence: float
    reason: str
    matched_signals: Dict[str, Any]



# Helpers
def _load_contract_cards(path: Path = CONTRACTS_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing contract cards: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("contracts.json must be a list")
    for d in data:
        if "doc_id" not in d:
            raise ValueError("Each card must contain doc_id")
    return data


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _cards_for_prompt(cards: List[Dict[str, Any]], limit_chars_per_card: int = 650) -> str:
    """
    Compact view for the LLM: index + doc_id + pages + top_terms + snippet.
    """
    blocks: List[str] = []
    for i, c in enumerate(cards, start=1):
        doc_id = c.get("doc_id", "")
        pages = c.get("pages", None)

        top_terms = ", ".join([str(x) for x in (c.get("top_terms") or [])])[:220]
        snippet = _normalize_ws((c.get("snippet") or "").replace("\r", ""))
        snippet = snippet[:limit_chars_per_card]

        blocks.append(
            f"[{i}] doc_id: {doc_id}\n"
            f"pages: {pages}\n"
            f"top_terms: {top_terms}\n"
            f"snippet: {snippet}\n"
        )
    return "\n\n".join(blocks)


def _safe_json_load(s: str) -> Optional[dict]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _coerce_indices(x: Any) -> List[int]:
    """
    Accept [1,2] or ["1","2"] or "1,2" etc.
    """
    if x is None:
        return []
    if isinstance(x, list):
        out: List[int] = []
        for v in x:
            try:
                out.append(int(str(v).strip()))
            except Exception:
                continue
        return out
    if isinstance(x, str):
        nums = re.findall(r"\d+", x)
        return [int(n) for n in nums]
    if isinstance(x, (int, float)):
        return [int(x)]
    return []

# Scope decision

def decide_query_scope(
    question: str,
    llm_model: str = "gpt-4o-mini",
) -> ScopeResult:
    """
    Decide if the question should consider:
    - single contract/company/document ("single")
    - multiple contracts/documents ("multi")

    Note: no temperature param here (to avoid unexpected-arg issues).
    """
    system = (
        "You are a classifier for a contract Q&A system.\n"
        "Decide whether the user's question requires looking at ONE contract/document\n"
        "or MULTIPLE contracts/documents.\n"
        "Return STRICT JSON only."
    )

    user = f"""
Question:
{question}

Return STRICT JSON only:
{{
  "scope": "single" or "multi",
  "confidence": 0.0,
  "reason": "short reason"
}}

Guidelines:
- single: mentions a specific doc name, contract number, company/supplier, or clearly refers to one contract.
- multi: asks to compare/summarize across contracts, uses plurals like "v zmluvách", "viaceré", "porovnaj", "najčastejšie", etc.
- If unclear, prefer "single".
""".strip()

    llm = ChatOpenAI(
        model=llm_model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    raw = getattr(resp, "content", "") or ""
    data = _safe_json_load(raw)

    if not data:
        return ScopeResult(
            scope="single",
            used_llm=True,
            confidence=0.0,
            reason="LLM returned non-JSON output, defaulting to single.",
            matched_signals={"llm_raw": raw},
        )

    scope = str(data.get("scope", "single")).strip().lower()
    if scope not in ("single", "multi"):
        scope = "single"

    try:
        conf = float(data.get("confidence", 0.6))
    except Exception:
        conf = 0.6
    conf = max(0.0, min(conf, 1.0))

    reason = str(data.get("reason", "")).strip()[:220]

    return ScopeResult(
        scope=scope,
        used_llm=True,
        confidence=conf,
        reason=reason or "Classified scope based on question wording.",
        matched_signals={"llm_json": data, "llm_raw": raw},
    )


# Core LLM routing

def llm_route(
    cards: List[Dict[str, Any]],
    question: str,
    max_doc_ids: int = 3,
    model: str = "gpt-4o-mini",
) -> RouteResult:
    prompt_cards = _cards_for_prompt(cards, limit_chars_per_card=650)

    system = (
        "You are a routing assistant for contract Q&A.\n"
        "Choose which contract document(s) the user question refers to,\n"
        "using ONLY the provided contract cards.\n"
        "Return STRICT JSON only."
    )

    user = f"""
User question:
{question}

Contract cards:
{prompt_cards}

Return STRICT JSON ONLY in this schema:
{{
  "card_indices": [1, 2],
  "confidence": 0.0,
  "reason": "short reason"
}}

Rules:
- card_indices are 1-based indices of the cards above.
- Pick ONLY cards that are clearly relevant to the question.
- Pick 1 card if the question is about a single contract/company.
- Pick up to {max_doc_ids} cards only if the question genuinely spans multiple contracts OR is ambiguous.
- Output MUST be valid JSON only (no markdown, no extra text).
""".strip()

    llm = ChatOpenAI(
        model=model,
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
    )
    resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])

    raw = getattr(resp, "content", "") or ""
    data = _safe_json_load(raw)

    if not data:
        return RouteResult(
            doc_ids=[],
            used_llm=True,
            confidence=0.0,
            reason="LLM returned non-JSON output.",
            matched_signals={"llm_raw": raw},
        )

    indices = _coerce_indices(data.get("card_indices"))
    indices = [i for i in indices if 1 <= i <= len(cards)]
    indices = indices[:max_doc_ids]

    doc_ids = [cards[i - 1]["doc_id"] for i in indices] if indices else []

    try:
        conf = float(data.get("confidence", 0.6))
    except Exception:
        conf = 0.6
    conf = max(0.0, min(conf, 1.0))

    reason = str(data.get("reason", "")).strip()[:220]

    return RouteResult(
        doc_ids=doc_ids,
        used_llm=True,
        confidence=conf if doc_ids else min(conf, 0.2),
        reason=reason or "LLM selected cards based on the contract snippets.",
        matched_signals={"llm_json": data, "llm_raw": raw},
    )


def route_contracts(
    question: str,
    max_doc_ids: int = 3,
    prefer_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
) -> RouteResult:
    """
    Backwards-compatible: routes to up to max_doc_ids docs.
    """
    cards = _load_contract_cards()

    if prefer_llm:
        return llm_route(cards, question, max_doc_ids=max_doc_ids, model=llm_model)

    return RouteResult(
        doc_ids=[],
        used_llm=False,
        confidence=0.0,
        reason="LLM routing disabled.",
        matched_signals={},
    )


def route_query(
    question: str,
    single_max_docs: int = 2,
    multi_max_docs: int = 6,
    prefer_llm: bool = True,
    llm_model: str = "gpt-4o-mini",
) -> RouteResult:
    """
    New API used by QA:
    - Decide scope (single vs multi) with decide_query_scope()
    - Route docs with llm_route()
    - If scope=single -> hard cap to 1 doc (avoid mixing)
    - If scope=multi  -> allow up to multi_max_docs
    """
    cards = _load_contract_cards()

    if not prefer_llm:
        return RouteResult(
            doc_ids=[],
            used_llm=False,
            confidence=0.0,
            reason="LLM routing disabled.",
            matched_signals={},
        )

    scope = decide_query_scope(question, llm_model=llm_model)

    # Ask LLM for doc picks (cap depends on scope)
    max_docs = multi_max_docs if scope.scope == "multi" else max(single_max_docs, 1)
    res = llm_route(cards, question, max_doc_ids=max_docs, model=llm_model)

    # Hardening: if scope is single, enforce 1 doc_id max
    if scope.scope == "single" and len(res.doc_ids) >= 1:
        res.doc_ids = res.doc_ids[:1]
        # enrich signals
        res.matched_signals["scope"] = asdict(scope)
        return res

    # If multi, cap to multi_max_docs (already capped, but safe)
    if scope.scope == "multi" and len(res.doc_ids) >= 1:
        res.doc_ids = res.doc_ids[:multi_max_docs]
        res.matched_signals["scope"] = asdict(scope)
        return res

    # If LLM returned none, keep empty: QA will fall back to global retrieval
    res.matched_signals["scope"] = asdict(scope)
    return res


def asdict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


if __name__ == "__main__":
    q = input("Zadejte otázku: ").strip()
    s = decide_query_scope(q, llm_model="gpt-4o-mini")
    print("\nSCOPE:", s.scope, "conf:", s.confidence, "reason:", s.reason)

    res = route_query(q, single_max_docs=2, multi_max_docs=6, prefer_llm=True, llm_model="gpt-4o-mini")
    print("\nROUTE RESULT")
    print("doc_ids:", res.doc_ids)
    print("used_llm:", res.used_llm)
    print("confidence:", res.confidence)
    print("reason:", res.reason)
    print("matched_signals:", json.dumps(res.matched_signals, ensure_ascii=False, indent=2))
