import os
import numpy as np
from sentence_transformers import SentenceTransformer


def chunk_text(text: str, chunk_chars: int = 400, overlap: int = 80):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_chars, text_length)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end == text_length:
            break

    return chunks


def infer_category(filename: str) -> str:
    # category = prefix before first "-" (fallback: "general")
    base = os.path.splitext(filename)[0]
    if "-" in base:
        return base.split("-", 1)[0].strip().lower() or "general"
    return "general"


def load_runbooks(path: str):
    runbooks = []
    for file in sorted(os.listdir(path)):
        if file.endswith(".md") or file.endswith(".txt"):
            with open(os.path.join(path, file), "r", encoding="utf-8") as f:
                runbooks.append(
                    {
                        "name": file,
                        "category": infer_category(file),
                        "text": f.read(),
                    }
                )
    return runbooks


def build_index(runbooks):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    chunk_records = []
    all_text_chunks = []

    for rb in runbooks:
        rb_chunks = chunk_text(rb["text"], chunk_chars=400, overlap=80)
        for i, ch in enumerate(rb_chunks):
            chunk_records.append(
                {
                    "runbook": rb["name"],
                    "category": rb["category"],
                    "chunk_id": i,
                    "text": ch,
                }
            )
            all_text_chunks.append(ch)

    emb = model.encode(all_text_chunks).astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    emb_norm = emb / norms

    return emb_norm, chunk_records, model

