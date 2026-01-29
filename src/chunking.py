from __future__ import annotations


def chunk_text(text, max_chars=1200, overlap=200):
    """
    Jednoduché chunkovanie po znakoch s overlapom.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == n:
            break

        start = max(0, end - overlap)

    return chunks
