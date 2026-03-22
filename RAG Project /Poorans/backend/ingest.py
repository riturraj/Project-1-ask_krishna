"""
ingest.py — Load Puranic texts, chunk, embed, and store in ChromaDB.
Run this ONCE (or re-run to refresh) whenever you add new text files.

Usage:
    python ingest.py
"""

import os
import glob
import chromadb
import os
import glob
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌  OPENAI_API_KEY not set in .env")

client_ai = OpenAI(api_key=OPENAI_API_KEY)

PURANAS_DIR  = os.path.join(os.path.dirname(__file__), "data", "puranas")
CHROMA_PATH  = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION   = "puranas"
CHUNK_WORDS  = 450
OVERLAP_WORDS = 100

# ── TEXT CHUNKER ──────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_WORDS])
        if chunk.strip():
            chunks.append(chunk)
        i += CHUNK_WORDS - OVERLAP_WORDS
    return chunks


# ── EMBED A SINGLE TEXT ───────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    response = client_ai.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


# ── MAIN INGEST ───────────────────────────────────────────────────────────────

def main():
    # Collect all .txt files recursively
    pattern = os.path.join(PURANAS_DIR, "**", "*.txt")
    txt_files = glob.glob(pattern, recursive=True)
    if not txt_files:
        print(f"\n❌  No .txt files found in {PURANAS_DIR}")
        print("    Please add your Puranic text files there and re-run.")
        return

    # Fresh ChromaDB collection
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION)
        print("♻️  Cleared existing collection.")
    except Exception:
        pass
    collection = client.create_collection(COLLECTION)

    total_chunks = 0
    doc_id = 0

    for filepath in txt_files:
        source_name = (
            os.path.basename(filepath)
            .replace(".txt", "")
            .replace("_", " ")
            .title()
        )
        print(f"\n📖  Processing: {source_name}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        print(f"    → {len(chunks)} chunks created")

        for idx, chunk in enumerate(chunks):
            vec = embed(chunk)
            collection.add(
                ids=[f"doc_{doc_id}"],
                embeddings=[vec],
                documents=[chunk],
                metadatas=[{"source": source_name, "chunk_index": idx}],
            )
            doc_id += 1
            total_chunks += 1

            # Progress indicator every 10 chunks
            if (idx + 1) % 10 == 0:
                print(f"    ↳ {idx + 1}/{len(chunks)} chunks embedded…")

    print(f"\n✅  Done! Ingested {total_chunks} chunks from {len(txt_files)} file(s) into ChromaDB.")
    print(f"    ChromaDB stored at: {os.path.abspath(CHROMA_PATH)}")


if __name__ == "__main__":
    main()
