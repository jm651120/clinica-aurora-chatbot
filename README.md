# Clínica Aurora — RAG Chatbot

A production-ready **Retrieval-Augmented Generation (RAG)** chatbot for a Portuguese aesthetic clinic, built entirely on the free tier. Serves as a business demo for applying RAG to Portuguese SMEs.

**Live demo:** https://clinica-aurora-chatbot-ad3ipttduedhfnzm7mdl9x.streamlit.app/

---

## Architecture

```
User query
    │
    ▼
[Query Rewriting]               Groq LLM expands colloquial PT terms to
    │                           clinical vocabulary before embedding
    ▼
[HuggingFace Embeddings]        sentence-transformers/all-MiniLM-L6-v2
    │                           runs fully offline on CPU (~50 ms/query)
    ▼
[ChromaDB]  ──── cosine similarity ──→  top-6 chunks
    │             (committed to repo for instant cold start)
    │
    ▼
[Pack Augmentation]             if query involves a pack comparison, prepend
    │                           a pre-computed, authoritative price block —
    │                           the LLM formats facts, never calculates them
    ▼
[Groq API]  ──── Llama 3.1 8B Instant ──→  grounded answer
    │             free tier · ~1 s latency
    ▼
[Post-processor]                deterministic PT-PT normalization (regex)
    │                           + guardrail enforcement (hard-stop truncation)
    ▼
[Streamlit chat UI]  ──→  answer + optional source citations
```

Two of the pipeline stages are entirely deterministic — no LLM, no probability. The design principle: delegate to the LLM only what requires generative capability (composing grounded text); handle everything else in code.

---

## Why these choices

### Embedding model — `all-MiniLM-L6-v2`

We evaluated both this model and `paraphrase-multilingual-MiniLM-L12-v2` against a 12-case retrieval test suite. The English model won on 9 of 12 queries. Counter-intuitive result explained: the clinic documents use formal, structured Portuguese whose vocabulary maps closely to English training data. The multilingual model spreads its 384-dimension capacity across 50+ languages, diluting quality on formal text. Practical takeaway: **for structured business documents, an English model often outperforms multilingual on non-English content**.

### Chunking — heading-aware, 500 chars / 50 overlap

`RecursiveCharacterTextSplitter` is configured with `\n## ` and `\n### ` as high-priority separators, so it always tries to break at Markdown section boundaries first. Without this, small sections at the end of a document (e.g. *Verniz de Gel*) collapse into the next section (e.g. *Pack Noiva*), diluting retrieval precision for both. The 50-character overlap prevents answers from being split across chunk boundaries.

### Generation — Groq SDK directly

`langchain-groq` was removed after discovering an httpx API incompatibility (the package passes a `proxies` kwarg that httpx ≥ 0.28 dropped). The `groq` Python SDK is used directly — simpler, faster, and one fewer abstraction layer.

### Pack comparison — pre-computed context injection

Pack vs. avulso questions have exactly one correct answer. Testing showed that Llama 3.1 8B reliably produced wrong results: inverting the comparison direction and computing incorrect savings figures. This is a deterministic problem — delegating it to a probabilistic model is the wrong tool. The fix: `src/pack_comparison.py` hardcodes the authoritative numbers and injects a structured `╔══ DADOS CALCULADOS ══╗` block at the top of the retrieval context when a pack query is detected. The LLM's only job is to format and present pre-verified numbers.

### Output post-processing — PT-PT normalization

Llama 3.1 8B was trained on a corpus dominated by Brazilian Portuguese. Despite explicit system-prompt instructions, it reliably drifts toward "você" and other BR-PT forms in conversational contexts — this is a training-data phenomenon that prompt instructions cannot fully override. The fix: `src/postprocessing.py` applies a deterministic regex pass after every generation call. The same module enforces a hard stop on social-engineering guardrail responses, truncating any content that bleeds through after the refusal sentence.

The architectural lesson: **prompt instructions fail predictably for small-vocabulary normalization problems where the model's weights disagree with the instruction**. Code is the right layer for those fixes.

### ChromaDB committed to git

For Streamlit Community Cloud deployment, the vector store (`chroma_db/`) is committed to the repository. This avoids a ~30–60 s cold start on the free tier while the embedding model downloads and re-embeds all documents on every dyno restart. The trade-off is ~5 MB added to the repo — acceptable for a demo project.

---

## Project structure

```
clinica-aurora-chatbot/
├── data/                       # Clinic knowledge base (edit these to customise)
│   ├── faq.md                  # 21 Q&As — treatments, age, pregnancy, booking, glossary
│   ├── tratamentos.md          # Full treatment catalogue with descriptions and prices
│   ├── precos.md               # Dedicated price table — all services in one dense file
│   ├── politicas.md            # Booking, cancellation, payment, RGPD, emergency policy
│   └── contactos.md            # Location, hours, team, accessibility
├── src/
│   ├── ingestion.py            # Load → chunk → embed → persist ChromaDB
│   ├── retrieval.py            # Similarity search + context formatter
│   ├── generation.py           # Groq API call + RAG prompt
│   ├── pack_comparison.py      # Pre-computed pack data + query detection
│   └── postprocessing.py       # PT-PT normalization + guardrail enforcement
├── tests/
│   ├── test_retrieval.py       # 12-case retrieval quality suite (scores + sources)
│   └── test_generation.py     # 7-case end-to-end suite + interactive REPL
├── chroma_db/                  # Pre-built vector store (committed — do not delete)
├── app.py                      # Streamlit chat interface
├── requirements.txt
├── .env.example
└── README.md
```

---

## Running locally

### 1. Clone and set up a virtual environment

```bash
git clone <your-repo-url>
cd clinica-aurora-chatbot
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

First run downloads `all-MiniLM-L6-v2` (~80 MB) into the HuggingFace cache. Subsequent runs are instant.

### 3. Configure your API key

```bash
cp .env.example .env
# open .env and paste your GROQ_API_KEY
```

Get a free key at [console.groq.com](https://console.groq.com) → API Keys. The free tier provides 14,400 requests/day and 6,000 tokens/min on Llama 3.1 8B.

### 4. Launch

```bash
streamlit run app.py
```

The `chroma_db/` vector store is already committed and loads instantly. You do **not** need to run ingestion unless you edit the documents in `data/`.

### Re-ingesting after document changes

```bash
python -m src.ingestion
```

This rebuilds `chroma_db/` from scratch. Commit the updated folder before deploying.

---

## Running the test suites

```bash
# Retrieval quality (no API key needed)
python tests/test_retrieval.py

# End-to-end generation (requires GROQ_API_KEY)
python tests/test_generation.py

# Interactive chat in the terminal
python tests/test_generation.py --interactive

# Single ad-hoc query
python tests/test_generation.py --query "Quanto custa o botox?"
```

---

## Deploying to Streamlit Community Cloud

1. Push the repository to GitHub (including `chroma_db/`):

   ```bash
   git push origin main
   ```

2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**

3. Select your repository, branch `main`, and main file `app.py`

4. Under **Advanced settings → Secrets**, add:

   ```toml
   GROQ_API_KEY = "gsk_..."
   ```

5. Click **Deploy**. The app starts in under 60 seconds.

> **Note:** Do not add `chroma_db/` to `.gitignore`. It must be present in the repo for the Cloud deployment to work without a cold-start ingestion step.

---

## Adapting to a different business

The knowledge base is entirely in `data/`. To deploy this for a different SME:

1. Replace or edit the four `.md` files in `data/` with the new business's content
2. Run `python -m src.ingestion` to rebuild the vector store
3. Update the `SYSTEM_PROMPT` in `src/generation.py` with the new business name and persona
4. Commit and push

No other code changes required.

---

## Technical reference

| Parameter | Value | Rationale |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast (80 MB, ~50 ms CPU), outperforms multilingual on formal PT docs |
| Chunk size | 500 chars | Fits one complete Q&A or section without splitting |
| Chunk overlap | 50 chars | Prevents answers from being split at chunk boundaries |
| Chunk separators | `\n## `, `\n### `, `\n\n`, `\n`, … | Respects Markdown heading structure |
| Retrieval k | 6 | Wider net needed after query expansion shifts the semantic neighbourhood |
| LLM | `llama-3.1-8b-instant` via Groq | Fast, free, good factual grounding |
| LLM temperature | 0.2 | Factual over creative |
| Max tokens | 1,024 | Concise answers; prevents padding |
| Vector store | ChromaDB (local, persistent) | Zero config, no server, committed to repo |

## Known limitations

- **Conversational retrieval drift**: in long multi-turn conversations, a vague follow-up query (e.g. *"E quanto dura?"*) may retrieve irrelevant chunks because the retriever sees only the current turn, not the conversation history. The LLM compensates from parametric memory but this is a hallucination risk. Mitigation: the query rewriting step helps, but a full history-aware rewriter would be more robust.
- **Colloquial Portuguese**: the English embedding model handles formal clinic vocabulary and expanded queries well. Very obscure slang not covered by the rewriting examples may still underperform. The rewrite prompt can be extended with additional examples at any time.
- **Pack detection scope**: `pack_comparison.py` currently injects Pack Noiva data for any pack-comparison query. If the user asks specifically about the Pack Anti-Envelhecimento comparison, the injected data is still Pack Noiva. Extending detection to select the correct pack by name is straightforward — the data structure already supports it.
