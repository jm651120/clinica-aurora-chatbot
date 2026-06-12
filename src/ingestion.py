"""
ingestion.py — Loads clinic documents, chunks them, and stores in ChromaDB.

Run once (or whenever documents change):
    python -m src.ingestion
"""

import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PERSIST_DIR = ROOT_DIR / "chroma_db"

# ── Model config ───────────────────────────────────────────────────────────────
# all-MiniLM-L6-v2: fast (80 MB, ~50 ms/query on CPU), performs well even on
# formal Portuguese because clinic vocabulary maps closely to English training
# data. Outperforms the multilingual model on structured documents like these.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking config ────────────────────────────────────────────────────────────
# 500 tokens ≈ 3-5 short paragraphs — large enough to hold a complete Q&A pair
# but small enough for the retriever to stay precise.
# 50-token overlap prevents answers from being split across chunk boundaries.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents(data_dir: Path) -> list:
    """Load all .pdf, .txt, and .md files from data_dir."""
    docs = []
    loaders = {
        ".pdf": lambda p: PyPDFLoader(str(p)),
        ".txt": lambda p: TextLoader(str(p), encoding="utf-8"),
        ".md":  lambda p: TextLoader(str(p), encoding="utf-8"),
    }

    for file in sorted(data_dir.iterdir()):
        loader_fn = loaders.get(file.suffix.lower())
        if loader_fn is None:
            continue
        try:
            loaded = loader_fn(file).load()
            # Tag each chunk with its source file name for later citation
            for doc in loaded:
                doc.metadata["source"] = file.name
            docs.extend(loaded)
            print(f"  ✓ Loaded: {file.name} ({len(loaded)} page(s))")
        except Exception as e:
            print(f"  ✗ Failed to load {file.name}: {e}")

    print(f"\nTotal documents loaded: {len(docs)}")
    return docs


def chunk_documents(documents: list) -> list:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Markdown headings are tried first so each section stays intact.
        # Without this, a small section (e.g. "Verniz de Gel") and the next
        # section ("Pack Noiva") collapse into one chunk, diluting retrieval.
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks  "
          f"(chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def create_vectorstore(chunks: list, reset: bool = False) -> Chroma:
    """Embed chunks and persist them in ChromaDB."""
    if reset and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print("Existing vector store deleted.")

    print(f"\nLoading embedding model: {EMBED_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("Embedding chunks and writing to ChromaDB …")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="clinica_aurora",
    )
    print(f"Vector store persisted at: {PERSIST_DIR}")
    return vectorstore


def run_ingestion(reset: bool = True) -> None:
    print("=" * 50)
    print("  Clínica Aurora — Document Ingestion")
    print("=" * 50)

    documents = load_documents(DATA_DIR)
    if not documents:
        raise RuntimeError(f"No documents found in {DATA_DIR}. "
                           "Add .pdf, .txt, or .md files and try again.")

    chunks = chunk_documents(documents)
    create_vectorstore(chunks, reset=reset)

    print("\nIngestion complete. You can now run the Streamlit app.")


if __name__ == "__main__":
    run_ingestion()
