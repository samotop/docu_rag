from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Optional


# Paths / sys.path
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"

# aby fungovali importy, keď router.py / ďalšie sú v src/
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC_DIR))

# build artefakty, ktoré kontrolujeme
PAGES_JSONL = ROOT / "index" / "pages.jsonl"
CHUNKS_JSONL = ROOT / "index" / "chunks.jsonl"
FAISS_DIR = ROOT / "index" / "faiss_pdf"
FAISS_FILES = [FAISS_DIR / "index.faiss", FAISS_DIR / "index.pkl"]
CONTRACT_CARDS = ROOT / "data" / "processed" / "contracts.json"


# helpers
def has_openai_key() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def banner(msg: str) -> None:
    print("\n" + "=" * 80)
    print(msg)
    print("=" * 80)


def doing(msg: str) -> None:
    print(f"⏳ Práve sa robí: {msg}")


def skip(msg: str) -> None:
    print(f"✅ Preskakujem: {msg} (už existuje)")


def done(msg: str) -> None:
    print(f"✅ Hotovo: {msg}")


def ensure_parent_dirs() -> None:
    (ROOT / "index").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)


# ----------------- build steps -----------------
def ensure_ingest_and_chunk(force: bool = False) -> None:
    """
    Vytvorí index/pages.jsonl + index/chunks.jsonl (ak chýbajú).
    """
    if not force and PAGES_JSONL.exists() and CHUNKS_JSONL.exists():
        skip("ingest + chunkovanie")
        return

    banner("BUILD 1/3: ingest + chunkovanie")
    doing("ingest + chunkovanie (build_index.py)")

    from build_index import main as build_index_main

    build_index_main()

    if not PAGES_JSONL.exists() or not CHUNKS_JSONL.exists():
        raise RuntimeError(
            "build_index.py dobehol, ale chýbajú očakávané súbory:\n"
            f"- {PAGES_JSONL}\n"
            f"- {CHUNKS_JSONL}\n"
        )

    done("ingest + chunkovanie")


def ensure_contract_cards(force: bool = False) -> None:
    """
    Vytvorí data/processed/contracts.json (router contract cards).
    """
    if not force and CONTRACT_CARDS.exists():
        skip("contract cards (router)")
        return

    banner("BUILD 2/3: contract cards")
    doing("contract cards (build_contract_cards.py)")

    from build_contract_cards import build_contract_cards

    build_contract_cards()

    if not CONTRACT_CARDS.exists():
        raise RuntimeError(
            "build_contract_cards.py dobehol, ale chýba výstup:\n"
            f"- {CONTRACT_CARDS}\n"
        )

    done("contract cards")


def ensure_embeddings_and_faiss(force: bool = False) -> None:
    """
    Vytvorí FAISS index v index/faiss_pdf (ak chýba).
    """
    faiss_ok = all(p.exists() for p in FAISS_FILES)
    if not force and faiss_ok:
        skip("embeddings + FAISS index")
        return

    if not has_openai_key():
        raise RuntimeError(
            "Chýba OPENAI_API_KEY (v env alebo v .env). "
            "Na embeddings + QA je potrebný."
        )

    banner("BUILD 3/3: embeddings + FAISS")
    doing("embeddings + FAISS (src/embeddings.py -> build_faiss_index)")

    from src.embeddings import build_faiss_index

    build_faiss_index()

    faiss_ok = all(p.exists() for p in FAISS_FILES)
    if not faiss_ok:
        raise RuntimeError(
            "build_faiss_index() dobehol, ale chýbajú FAISS súbory:\n"
            f"- {FAISS_FILES[0]}\n"
            f"- {FAISS_FILES[1]}\n"
        )

    done("embeddings + FAISS index")


def build_if_needed(force: bool = False) -> None:
    """
    Hlavný build orchestrátor – spustí iba to, čo treba.
    """
    ensure_parent_dirs()
    banner("CHECK: build artefakty")

    # len informačné vypísanie, čo existuje / neexistuje
    print(f"- pages.jsonl:   {'OK' if PAGES_JSONL.exists() else 'MISSING'}  ({PAGES_JSONL})")
    print(f"- chunks.jsonl:  {'OK' if CHUNKS_JSONL.exists() else 'MISSING'} ({CHUNKS_JSONL})")
    print(f"- contracts.json:{'OK' if CONTRACT_CARDS.exists() else 'MISSING'} ({CONTRACT_CARDS})")
    print(
        f"- faiss index:   {'OK' if all(p.exists() for p in FAISS_FILES) else 'MISSING'} "
        f"({FAISS_FILES[0].parent})"
    )

    ensure_ingest_and_chunk(force=force)
    ensure_contract_cards(force=force)
    ensure_embeddings_and_faiss(force=force)

    banner("BUILD: všetko pripravené")
    print("✅ Pipeline je pripravená. Teraz môžete položiť otázku.")


# QA
def run_qa(
    question: str,
    top_k: int = 6,
    pool_k: int = 30,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
) -> None:
    if not has_openai_key():
        raise RuntimeError(
            "Chýba OPENAI_API_KEY (v env alebo v .env). "
            "Na QA (ChatOpenAI) je potrebný."
        )

    banner("QA")
    doing("hľadám odpoveď (retrieval + LLM)")

    from src.qa import answer_question

    out = answer_question(
        question,
        top_k=top_k,
        pool_k=pool_k,
        router_max_docs_single=2,
        router_max_docs_multi=6,
        model=model,
        temperature=temperature,
    )

    done("odpoveď vygenerovaná")

    print("\n" + (out.get("answer") or "").strip() + "\n")

    sources = out.get("sources") or []
    if sources:
        print("Zdroje:")
        for s in sources:
            print(
                f"[{s['n']}] {s['doc_id']} (strana {s['page']}), "
                f"chunk {s['chunk_id']}, method {s.get('method','')}"
            )
    else:
        print("Zdroje: (žiadne citácie v odpovedi)")


#  main
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Mini RAG pipeline: build (ak treba) -> QA")
    parser.add_argument("--question", "-q", type=str, default="", help="Otázka pre QA")
    parser.add_argument("--force", action="store_true", help="Prebuildne všetko nanovo")
    parser.add_argument("--top_k", type=int, default=6, help="Koľko chunkov poslať do LLM")
    parser.add_argument("--pool_k", type=int, default=30, help="Pool size pre retrieval (pred fúziou)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM model pre ChatOpenAI")
    parser.add_argument("--temperature", type=float, default=0.0, help="Teplota pre LLM")
    args = parser.parse_args(argv)

    # 1) build ešte pred otázkou
    build_if_needed(force=args.force)

    # 2) potom otázka
    question = (args.question or "").strip()
    if not question:
        question = input("\nZadajte otázku: ").strip()

    if not question:
        print("❌ Nebola zadaná žiadna otázka.")
        return 2

    # 3) QA
    run_qa(
        question=question,
        top_k=args.top_k,
        pool_k=args.pool_k,
        model=args.model,
        temperature=args.temperature,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())