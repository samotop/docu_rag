from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[0]


PAGES_JSONL = ROOT / "index" / "pages.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "contracts.json"


@dataclass
class ContractCard:
    doc_id: str
    pages: int
    snippet: str
    top_terms: List[str]


def read_jsonl(path: Path) -> List[dict]:
    items: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _normalize_text(t: str) -> str:
    t = t.replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _looks_like_ocr_garbage(t: str) -> bool:
    """
    Heuristika len na detekciu "šumu" (nie stopwords).
    - veľa ne-alfanumerických znakov
    - veľa veľmi krátkych tokenov
    """
    if not t:
        return True

    sample = t[:800]
    if len(sample) < 80:
        return True

    # pomer písmen (latin + diakritika) k ostatným
    letters = re.findall(r"[A-Za-zÁ-ž]", sample)
    ratio_letters = len(letters) / max(1, len(sample))
    if ratio_letters < 0.25:
        return True

    toks = re.findall(r"[A-Za-zÁ-ž0-9]{2,}", sample)
    if len(toks) < 10:
        return True

    return False


def _extract_terms(text: str, k: int = 12) -> List[str]:
    """
    Jednoduché 'keywords' bez slovníka stopwords:
    - berieme len tokeny dĺžky 4+
    - preferujeme tokeny s veľkým písmenom / skratky / čísla, ktoré sú typické pre zmluvy
    - vrátime unikátne v poradí výskytu
    """
    if not text:
        return []
    # tokeny: písmená/čísla/.-&_
    raw = re.findall(r"[A-Za-zÁ-ž0-9&._-]{4,}", text)
    seen = set()
    out: List[str] = []

    # preferenčné triedenie: acronyms/čísla/veľké písmená dopredu
    def score(tok: str) -> int:
        s = 0
        if tok.isupper() and len(tok) <= 10:
            s += 3
        if any(ch.isdigit() for ch in tok):
            s += 2
        if tok[:1].isupper():
            s += 1
        return -s  # menšie je lepšie pre sort (negatív)

    # zachovať poradie, ale s preferenciou "silných" tokenov:
    # pre-selection z prvých ~200 tokenov
    cand = raw[:250]
    cand_sorted = sorted(cand, key=score)

    for tok in cand_sorted:
        t = tok.strip("._-")
        if len(t) < 4:
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
        if len(out) >= k:
            break
    return out


def build_contract_cards(
    pages_jsonl: Path = PAGES_JSONL,
    out_path: Path = OUT_PATH,
    max_snippet_chars: int = 1200,
) -> Path:
    if not pages_jsonl.exists():
        raise FileNotFoundError(f"Missing pages.jsonl at: {pages_jsonl}")

    pages = read_jsonl(pages_jsonl)

    # group by doc_id -> pages list
    by_doc: Dict[str, List[dict]] = defaultdict(list)
    for p in pages:
        by_doc[p["doc_id"]].append(p)

    cards: List[ContractCard] = []

    for doc_id, plist in sorted(by_doc.items(), key=lambda kv: kv[0].lower()):
        plist_sorted = sorted(plist, key=lambda x: int(x.get("page", 0) or 0))
        total_pages = max(int(x.get("page", 0) or 0) for x in plist_sorted) if plist_sorted else 0

        # vyberame snippet z prvých 1–2 strán, ale preskočíme očividný OCR garbage
        snippet_parts: List[str] = []
        for p in plist_sorted[:4]:  # skúsime prvé 4 stránky
            txt = _normalize_text(str(p.get("text", "") or ""))
            if not txt:
                continue
            if _looks_like_ocr_garbage(txt):
                continue
            snippet_parts.append(txt)
            if sum(len(x) for x in snippet_parts) >= max_snippet_chars:
                break

        # fallback: ak všetko vyzerá ako garbage, zoberieme aspoň niečo z 1. stránky (aj keby šum)
        if not snippet_parts and plist_sorted:
            snippet_parts.append(_normalize_text(str(plist_sorted[0].get("text", "") or "")))

        snippet = "\n\n".join(snippet_parts).strip()
        snippet = snippet[:max_snippet_chars].strip()

        top_terms = _extract_terms(snippet, k=12)

        cards.append(
            ContractCard(
                doc_id=doc_id,
                pages=total_pages,
                snippet=snippet,
                top_terms=top_terms,
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in cards], f, ensure_ascii=False, indent=2)

    print(f"✅ Wrote {len(cards)} contract cards to {out_path}")
    return out_path


if __name__ == "__main__":
    build_contract_cards()