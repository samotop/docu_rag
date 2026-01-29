from pathlib import Path
import os
import json
from dotenv import load_dotenv

from src.ingest import load_pdfs_from_folder, write_pages_jsonl, summarize_pages
from src.chunking import chunk_text

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def write_chunks_jsonl(chunks, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")


def main():
    data_dir = ROOT / "data" / "raw"
    out_pages = ROOT / "index" / "pages.jsonl"
    out_chunks = ROOT / "index" / "chunks.jsonl"

    poppler_path = os.getenv("POPPLER_PATH")
    print("POPPLER_PATH =", poppler_path)

    pages = load_pdfs_from_folder(
        data_dir,
        # DOCX -> PDF replace (NEW)
        convert_docx_to_pdf=True,
        replace_docx=True,  # docx sa po konverzii zmaže

        # extraction
        ocr_enabled=True,
        ocr_threshold_chars=150,  # odporúčam vyššie, aby OCR neskákalo zbytočne
        ocr_lang=os.getenv("OCR_LANG", "ces"),
        ocr_dpi=300,
        poppler_path=poppler_path,
    )

    write_pages_jsonl(pages, out_pages)

    # chunky
    all_chunks = []
    chunk_counter = 0

    for p in pages:
        pieces = chunk_text(p.text, max_chars=1200, overlap=200)
        for t in pieces:
            chunk_counter += 1
            all_chunks.append(
                {
                    "chunk_id": f"c{chunk_counter:05d}",
                    "doc_id": p.doc_id,
                    "page": p.page,
                    "method": p.method,
                    "text": t,
                }
            )

    write_chunks_jsonl(all_chunks, out_chunks)

    stats = summarize_pages(pages)
    print("✅ Done")
    print("Pages:", stats["total_pages"])
    print("Empty pages:", stats["empty_pages"])
    print("Method counts:", stats["method_counts"])
    print("Chunks:", len(all_chunks))
    print("By doc:")
    for doc, s in stats["by_doc"].items():
        print(f" - {doc}: {s['pages']} pages, empty {s['empty_pages']}, ocr_pages {s['ocr_pages']}")


if __name__ == "__main__":
    main()
