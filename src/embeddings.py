import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# načítaj .env (OPENAI_API_KEY + ostatné env)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CHUNKS_PATH = ROOT / "index" / "chunks.jsonl"
INDEX_DIR = ROOT / "index" / "faiss_pdf"


def build_faiss_index():
    # 1) load chunks (JSONL)
    chunks = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    print("Loaded chunks:", len(chunks), "from", CHUNKS_PATH)

    # 2) embedder
    embedder = OpenAIEmbeddings(model="text-embedding-3-small")

    # 3) docs (LangChain Document + metadata)
    docs = []
    for ch in chunks:
        doc = Document(
            page_content=ch["text"],
            metadata={
                "chunk_id": ch["chunk_id"],
                "doc_id": ch["doc_id"],     # filename
                "page": ch["page"],         # 1-based
                "method": ch.get("method", "unknown"),  # pdfplumber / ocr
            },
        )
        docs.append(doc)

    # 4) build + save FAISS
    db = FAISS.from_documents(docs, embedder)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db.save_local(str(INDEX_DIR))

    print("Saved index to:", INDEX_DIR)
    return INDEX_DIR


if __name__ == "__main__":
    build_faiss_index()