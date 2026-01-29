from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import os
import re
import pdfplumber


@dataclass
class PageText:
    doc_id: str
    page: int
    text: str
    method: str


# DOCX -> PDF (replace)

def convert_docx_to_pdf_replace(folder: Path) -> None:
    """
    Nájde *.docx v priečinku, skonvertuje na *.pdf do toho istého priečinka
    a pôvodný *.docx vymaže (replace).
    """
    docx_files = sorted(folder.glob("*.docx"))
    if not docx_files:
        return

    try:
        from docx2pdf import convert  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Chýba knižnica docx2pdf. Nainštalujte: pip install docx2pdf\n"
            f"Detail: {e}"
        )

    for docx_path in docx_files:
        pdf_path = docx_path.with_suffix(".pdf")

        # ak PDF už existuje, berieme ho ako cieľ a docx zmažeme (aby sa už nepoužil)
        if pdf_path.exists():
            try:
                docx_path.unlink()
            except PermissionError:
                raise PermissionError(
                    f"Nemôžem vymazať {docx_path}. Zavrite Word / Explorer náhľad a skúste znova."
                )
            continue

        # convert(in, out) - out môže byť file alebo folder
        try:
            convert(str(docx_path), str(pdf_path))
        except Exception as e:
            raise RuntimeError(
                f"Nepodarilo sa konvertovať DOCX -> PDF:\n"
                f"  DOCX: {docx_path}\n  PDF:  {pdf_path}\n"
                f"Tip: na Windows musí byť nainštalovaný MS Word.\n"
                f"Detail: {e}"
            )

        # po úspechu zmažeme DOCX
        try:
            docx_path.unlink()
        except PermissionError:
            raise PermissionError(
                f"PDF sa vytvorilo, ale nemôžem vymazať {docx_path}. "
                f"Zavrite Word / Explorer náhľad a skúste znova."
            )


# Text cleanup + extraction

def clean_text(t: Optional[str]) -> str:
    if not t:
        return ""
    t = t.replace("\r", "\n")
    lines = [ln.strip() for ln in t.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def extract_page_text_pdfplumber(page) -> str:
    """
    Robust pdfplumber extraction:
    1) extract_text()
    2) fallback extract_words() -> join
    """
    text = clean_text(page.extract_text(x_tolerance=2, y_tolerance=2))
    if text:
        return text

    words = page.extract_words(x_tolerance=2, y_tolerance=2)
    if not words:
        return ""

    joined = " ".join(w.get("text", "") for w in words if w.get("text"))
    return clean_text(joined)


def text_quality_score(text: str) -> float:
    """
    Hrubé skóre 0..1: koľko je v texte "normálnych" písmen/čísiel.
    Vyššie = lepšie.
    """
    if not text:
        return 0.0
    text = text.strip()
    n = len(text)
    if n == 0:
        return 0.0

    letters = sum(ch.isalpha() for ch in text)
    alnum = sum(ch.isalnum() for ch in text)
    printable = sum(ch.isprintable() for ch in text)

    letters_ratio = letters / n
    alnum_ratio = alnum / n
    printable_ratio = printable / n

    words = re.findall(r"[A-Za-zÁ-ž0-9]{3,}", text)
    word_count = len(words)

    score = 0.45 * letters_ratio + 0.45 * alnum_ratio + 0.10 * printable_ratio
    if word_count >= 10:
        score += 0.10
    return min(score, 1.0)


def is_gibberish(text: str, threshold: float = 0.50) -> bool:
    text = (text or "").strip()
    if len(text) < 30:
        return True
    return text_quality_score(text) < threshold


def ocr_page_text(
    pdf_path: Path,
    page_number_1based: int,
    lang: str = "ces",
    dpi: int = 300,
    poppler_path: Optional[str] = None,
) -> str:
    """
    OCR fallback (ak PDF nemá text).
    .env odporúčanie:
      POPPLER_PATH=...
      TESSERACT_CMD=...
      OCR_LANG=ces
      TESSDATA_PREFIX=... (folder kde sú .traineddata)
    """
    from pdf2image import convert_from_path
    import pytesseract

    # tesseract.exe (ak nie je v PATH)
    tcmd = os.getenv("TESSERACT_CMD")
    if tcmd:
        pytesseract.pytesseract.tesseract_cmd = tcmd

    # tessdata folder (traineddata)
    tessdata = os.getenv("TESSDATA_PREFIX")
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata

    lang = os.getenv("OCR_LANG", lang)

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number_1based,
        last_page=page_number_1based,
        poppler_path=poppler_path,
    )
    img = images[0]

    config = "--oem 3 --psm 6"
    raw = pytesseract.image_to_string(img, lang=lang, config=config)
    text = clean_text(raw)

    # radšej nič ako nezmysel v indexe
    if is_gibberish(text):
        return ""

    return text


# Main loader
def load_pdfs_from_folder(
    folder: Path,
    ocr_enabled: bool = True,
    ocr_threshold_chars: int = 50,
    ocr_lang: str = "ces",
    ocr_dpi: int = 300,
    poppler_path: Optional[str] = None,
    convert_docx_to_pdf: bool = True,
    replace_docx: bool = True,
) -> List[PageText]:
    """
    Načíta všetky PDF z folderu.
    NEW:
    - ak sú v folderi DOCX, prekonvertuje ich na PDF a (ak replace_docx=True) vymaže DOCX.

    PDF extrakcia:
    - Najprv pdfplumber
    - Keď je text prázdny / krátky, spraví OCR (ak ocr_enabled=True)
    """
    folder = Path(folder)

    if convert_docx_to_pdf:
        convert_docx_to_pdf_replace(folder)


    pdf_files = sorted(folder.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Nenašiel som žiadne PDF v: {folder}")

    pages: List[PageText] = []

    for pdf_path in pdf_files:
        doc_id = pdf_path.name
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i0, page in enumerate(pdf.pages):
                page_num = i0 + 1

                text = extract_page_text_pdfplumber(page)
                method = "pdfplumber"

                if ocr_enabled and len(text) < ocr_threshold_chars:
                    ocr_text = ocr_page_text(
                        pdf_path,
                        page_number_1based=page_num,
                        lang=ocr_lang,
                        dpi=ocr_dpi,
                        poppler_path=poppler_path,
                    )
                    if len(ocr_text) > len(text):
                        text = ocr_text
                        method = "ocr"

                pages.append(PageText(doc_id=doc_id, page=page_num, text=text, method=method))

    return pages


def write_pages_jsonl(pages: List[PageText], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")


def summarize_pages(pages: List[PageText]) -> Dict[str, Any]:
    total = len(pages)
    empty = sum(1 for p in pages if not p.text.strip())

    by_doc: Dict[str, Dict[str, Any]] = {}
    method_counts: Dict[str, int] = {"pdfplumber": 0, "ocr": 0}

    for p in pages:
        method_counts[p.method] = method_counts.get(p.method, 0) + 1
        d = by_doc.setdefault(
            p.doc_id,
            {"pages": 0, "empty_pages": 0, "empty_page_numbers": [], "ocr_pages": 0},
        )
        d["pages"] += 1
        if p.method == "ocr":
            d["ocr_pages"] += 1
        if not p.text.strip():
            d["empty_pages"] += 1
            d["empty_page_numbers"].append(p.page)

    return {
        "total_pages": total,
        "empty_pages": empty,
        "method_counts": method_counts,
        "by_doc": by_doc,
    }