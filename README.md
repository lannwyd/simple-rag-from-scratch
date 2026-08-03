# RAG From Scratch — CLI

A minimal Retrieval-Augmented Generation (RAG) pipeline built **from scratch in Python**, with no LangChain or other RAG frameworks. Built as a learning project to understand what's actually happening under the hood before using higher-level abstractions.

## What it does

Ask a question about a text file, and get an answer grounded in that file's content — not the LLM's general knowledge.

Pipeline: `chunk → embed → retrieve → generate`

1. **Chunking** — splits the source text into sentence-grouped chunks (~500 chars each), rather than naive fixed-size splitting.
2. **Embedding** — converts each chunk into a 384-dim vector locally using `sentence-transformers` (`all-MiniLM-L6-v2`). No API calls, no rate limits.
3. **Retrieval** — embeds the user's question, computes cosine similarity against every chunk, returns the top-k most relevant.
4. **Generation** — feeds the retrieved chunks + question to an LLM (via Groq API) as context, so the answer is grounded in the source text.

## Stack

- Python
- `sentence-transformers` — local embeddings
- `numpy` — cosine similarity (hand-implemented, no vector DB yet)
- `groq` — LLM inference
- `python-dotenv` — API key management

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install sentence-transformers groq python-dotenv numpy
```

Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```

Put your source text in `data/data.txt`.

## Usage

```bash
python main.py
```

You'll be prompted to ask a question. The script retrieves the most relevant chunks from `data/data.txt` and streams back an answer grounded in them.

## Known limitations

- **Sentence splitting is naive** (`text.split(". ")`) — breaks on abbreviations (e.g. "R.R." gets treated as a sentence end). A proper tokenizer (`nltk`/`spacy`) would fix this.
- **No overlap between chunks** — context right at a chunk boundary can be lost.
- **No vector database** — embeddings are recomputed on every run and held in memory as a plain list. Fine for small files, doesn't scale.
- **Retrieval is brute-force** — cosine similarity is computed against every chunk on every query; no indexing.

## What's next

- [ ] Fix sentence splitting (regex or real tokenizer)
- [ ] Add chunk overlap
- [ ] Swap in-memory retrieval for a vector DB (Chroma → Qdrant)
- [ ] Hybrid search (keyword + vector)
- [ ] FastAPI backend + Next.js frontend
- [ ] Dockerize and deploy

## Why build this by hand

Most RAG tutorials start with LangChain, which hides the exact mechanics this project is meant to teach: how chunking strategy affects retrieval, why cosine similarity is the right comparison, and what a vector database is actually solving. This version implements each step manually first, with abstractions (LangChain, Chroma) planned as later, deliberate additions — not starting points.
