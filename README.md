# KUSK RAG – mini nástroj na vyhľadávanie a Q&A nad 10 zmluvami

Tento projekt je jednoduchý **RAG (Retrieval-Augmented Generation)** pipeline:
- načíta dokumenty (PDF/DOCX) zo zložky `data/raw/`,
- rozdelí ich na časti (chunky) a zaindexuje (BM25 + FAISS),
- umožní položiť otázku v prirodzenom jazyku a vráti odpoveď **iba** na základe obsahu dokumentov,
- ku každej odpovedi vráti citácie (názov súboru + číslo strany + chunk).

> ⚠️ Prosím vložte **10 zmlúv** do priečinka `data/raw/`.
> Ideálne používajte **neoskenované (digitálne) PDF** – kvalita vyhľadávania a citácií je výrazne lepšia než pri skenoch (OCR je podporované, ale menej presné).

---

## 1) Požiadavky

- Python 3.10+
- `pip install -r requirements.txt`
- OpenAI API key v `.env` alebo env premenných (pre embeddings aj odpovede)

Príklad `.env`:
```env
OPENAI_API_KEY=...  # vložte vlastný kľúč
```

---

## 2) OCR pre skeny (voliteľné)

Ak máte v `data/raw/` **oskenované PDF**, pipeline použije OCR cez **Tesseract**.
Pre digitálne (textové) PDF OCR netreba.

### Inštalácia systémových nástrojov

**Windows:**
- Tesseract: nainštalujte `Tesseract-OCR`
- Poppler: stiahnite a rozbaľte Poppler (kvôli `pdf2image`)

**macOS:**
```bash
brew install tesseract poppler
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr poppler-utils
```

---

## 3) Nastavenie `.env` – dôležité (cesty sú lokálne!)

V projekte môžete mať okrem `OPENAI_API_KEY` aj tieto premenné:

```env
POPPLER_PATH=D:\poppler\Library\bin
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
OCR_LANG=ces
TESSDATA_PREFIX=D:\tessdata
```

> ⚠️ **Tieto cesty sú vždy špecifické pre konkrétny počítač.**  
> Ak projekt spúšťate na inom stroji, **nahraďte ich svojimi cestami** alebo ich odstráňte, ak OCR nepotrebujete.

### Čo znamenajú premenné
- `POPPLER_PATH` – cesta ku `pdftoppm` / Poppler binárom (Windows, pre `pdf2image`)
- `TESSERACT_CMD` – cesta k `tesseract.exe` (Windows)
- `OCR_LANG` – jazyk pre OCR (napr. `ces`, `eng`, `slk`)
- `TESSDATA_PREFIX` – cesta k priečinku s OCR jazykovými modelmi (`.traineddata`)

### Rýchle overenie (Windows)
V PowerShelli:
```powershell
tesseract --version
```
Ak to nefunguje, pravdepodobne nie je v PATH – vtedy je dôležité mať správne `TESSERACT_CMD`.

---

## 4) Rýchly štart

### A) Pripravte dokumenty
1. Vložte **10 dokumentov** (PDF alebo DOCX) do:
   - `data/raw/`

### B) Spustite aplikáciu
Najjednoduchšie je spustiť `app.py`, ktorý:
- ak chýbajú indexy, tak všetko sám vybuduje (ingest/chunky/embeddings),
- potom sa spýta na otázku a odpovie.

```bash
python app.py
```

Alebo priamo s otázkou:

```bash
python app.py --question "Aká je výška zmluvnej ceny?"
```

Voliteľné parametre:

```bash
python app.py --top_k 6 --pool_k 30 --model gpt-4o-mini --temperature 0.0
```

---

## 5) Ako to funguje (pipeline)

### Build (indexovanie)
Build sa skladá z 3 krokov:

1) **Ingest + chunkovanie**  
   - číta súbory z `data/raw/`
   - (ak je DOCX) konvertuje do PDF
   - z PDF vyťahuje text po stranách (ak treba OCR)
   - rozdelí text na chunky
   - výstupy:
     - `index/pages.jsonl`
     - `index/chunks.jsonl`

2) **Contract cards (router)**  
   - vytvorí jednoduché “karty” dokumentov (názov, stručný obsah, meta)
   - slúži na lepší routing otázok pri multi-doc dotazoch
   - výstup:
     - `data/processed/contracts.json`

3) **Embeddings + FAISS**  
   - spraví embeddings pre chunky
   - uloží FAISS index
   - výstup:
     - `index/faiss_pdf/index.faiss`
     - `index/faiss_pdf/index.pkl`

> `app.py` tieto kroky spraví automaticky iba vtedy, keď artefakty chýbajú (alebo pri `--force`).

### QA (otázka → odpoveď)
- retrieval: BM25 + FAISS (fúzia výsledkov)
- do LLM sa posiela top-k chunkov (limit kontextu)
- LLM musí odpovedať **iba** z poskytnutých chunkov a prikladať citácie `[1]`, `[2]`, ...

---

## 6) Logovanie otázok a odpovedí

Doporučené umiestnenie logov:
- `logs/qa_log.jsonl` alebo `logs/qa_log.csv`

---

## 7) Testovanie kvality odpovedí

Odporúčaný postup:
- pripravte 5–10 testovacích otázok,
- ku každej uveďte očakávaný fakt a kde sa nachádza (súbor + strana),
- overte, že odpoveď sedí a citácia smeruje správne.

---

## 8) Štruktúra projektu

```
.
├─ app.py
├─ build_index.py
├─ build_contract_cards.py
├─ requirements.txt
├─ data/
│  ├─ raw/                 # SEM vložte 10 zmlúv (PDF/DOCX)
│  └─ processed/
│     └─ contracts.json
├─ index/
│  ├─ pages.jsonl
│  ├─ chunks.jsonl
│  └─ faiss_pdf/
│     ├─ index.faiss
│     └─ index.pkl
└─ src/
   ├─ ingest.py
   ├─ chunking.py
   ├─ embeddings.py
   ├─ qa.py
   └─ router.py
```

---

## 9) Použité knižnice a modely

- Parsing PDF: `pdfplumber`
- OCR: `pytesseract`, `pdf2image` (+ systémový `tesseract`, `poppler`)
- Vector store: `faiss-cpu`
- Sparse retrieval: `rank-bm25`
- LLM + embeddings: `langchain-openai` (OpenAI)
- Konfigurácia env: `python-dotenv`
- Konverzia DOCX→PDF: `docx2pdf`
