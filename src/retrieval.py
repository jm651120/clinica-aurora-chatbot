"""
retrieval.py — Loads the ChromaDB vector store and retrieves relevant chunks.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

ROOT_DIR = Path(__file__).parent.parent
PERSIST_DIR = ROOT_DIR / "chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieve 6 chunks: wider net helps when query expansion shifts the semantic
# neighbourhood, and Llama 3.1 8B handles ~3 000 tokens of context fine.
DEFAULT_K = 6


def load_vectorstore() -> Chroma:
    """Load the persisted ChromaDB collection. Raises if ingestion hasn't run."""
    if not PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {PERSIST_DIR}. "
            "Run `python -m src.ingestion` first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings,
        collection_name="clinica_aurora",
    )


def retrieve(query: str, vectorstore: Chroma, k: int = DEFAULT_K) -> list[Document]:
    """Return the k most relevant document chunks for a query."""
    return vectorstore.similarity_search(query, k=k)


def retrieve_with_scores(
    query: str,
    vectorstore: Chroma,
    k: int = DEFAULT_K,
) -> list[tuple[Document, float]]:
    """
    Return (document, relevance_score) pairs, score in [0, 1] where 1 = perfect match.

    Uses similarity_search_with_relevance_scores which normalises the raw
    L2 distance to a cosine-equivalent scale (valid because embeddings are
    unit-normalised). Scores below ~0.4 usually indicate the query has no
    good match in the knowledge base.
    """
    return vectorstore.similarity_search_with_relevance_scores(query, k=k)


def format_context(docs: list[Document]) -> str:
    """Flatten retrieved chunks into a single numbered context string."""
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "desconhecido")
        parts.append(f"[{i}] (fonte: {source})\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)
