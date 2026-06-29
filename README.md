# Basic RAG Pipeline

A minimal Retrieval-Augmented Generation (RAG) system built with **LangChain**, **OpenAI**, and **ChromaDB**.

## What It Does

Query your own PDF documents using natural language. The system retrieves the most relevant passages and uses GPT-4o-mini to generate a grounded answer with source citations.

## Architecture

```
PDF Files
   │
   ▼
[1] PyPDFLoader          ← loads raw text + metadata (page number, source)
   │
   ▼
[2] RecursiveTextSplitter ← splits into 1000-char chunks, 200-char overlap
   │
   ▼
[3] OpenAI Embeddings     ← text-embedding-3-small converts chunks → vectors
   │
   ▼
[4] ChromaDB              ← stores vectors locally (persists on disk)
   │
   │   (at query time)
   │
[5] Retriever             ← cosine similarity search, returns top-4 chunks
   │
   ▼
[6] GPT-4o-mini           ← generates answer grounded in retrieved context
   │
   ▼
Answer + Source Citations
```

## Setup

```bash
# 1. Clone / navigate to this folder
cd basic-rag

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI key
cp .env.example .env
# Edit .env and paste your key

# 4. Add PDF files
mkdir docs
# Copy your PDFs into the docs/ folder

# 5. Run
python rag.py
```

## Key Concepts Demonstrated

| Concept | Where in Code |
|---|---|
| Document loading | `load_documents()` |
| Chunking strategy | `split_documents()` — `RecursiveCharacterTextSplitter` |
| Embedding | `build_vector_store()` — `text-embedding-3-small` |
| Vector store | ChromaDB with local persistence |
| Retrieval | `as_retriever(search_type="similarity", k=4)` |
| Prompt engineering | `RAG_PROMPT` — grounding + fallback instruction |
| QA chain | `RetrievalQA` with `chain_type="stuff"` |
| Source citations | `return_source_documents=True` |

## Tuning Parameters

| Parameter | Default | Effect |
|---|---|---|
| `CHUNK_SIZE` | 1000 | Larger = more context per chunk, but noisier |
| `CHUNK_OVERLAP` | 200 | Prevents context loss at chunk boundaries |
| `TOP_K` | 4 | More chunks = more context, but higher cost |
| `CHAT_MODEL` | gpt-4o-mini | Swap to gpt-4o for harder questions |

## What's Next (Advanced RAG)

- **Reranking** — add a Cohere reranker after retrieval for better precision
- **Hybrid search** — combine vector + BM25 keyword search
- **Eval with RAGAS** — measure faithfulness, answer relevance, context recall
- **Agentic RAG** — let the model decide when to retrieve vs answer from memory
