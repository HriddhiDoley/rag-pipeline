"""
Basic RAG Pipeline
==================
Stack: LangChain + OpenAI + ChromaDB
Supports: PDF files

Flow:
  1. Load PDF(s) from ./docs/
  2. Split into chunks
  3. Embed with OpenAI text-embedding-3-small
  4. Store in ChromaDB (local, persistent)
  5. Query with a question → retrieve top-K chunks → GPT-4o-mini answers
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()

DOCS_DIR   = "./docs"          # Put your PDFs here
CHROMA_DIR = "./chroma_db"     # Vector store persists here
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"

CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap between chunks (preserves context at boundaries)
TOP_K         = 4      # number of chunks to retrieve per query

# ── Prompt ────────────────────────────────────────────────────────────────────

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
)

# ── Step 1: Load PDFs ─────────────────────────────────────────────────────────

def load_documents():
    """Load all PDFs from the docs/ directory."""
    if not Path(DOCS_DIR).exists():
        raise FileNotFoundError(f"Docs folder not found: '{DOCS_DIR}'. Create it and add PDF files.")

    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()

    if not docs:
        raise ValueError(f"No PDFs found in '{DOCS_DIR}'. Add at least one PDF file.")

    print(f"[1/4] Loaded {len(docs)} page(s) from PDF(s)")
    return docs

# ── Step 2: Chunk ─────────────────────────────────────────────────────────────

def split_documents(docs):
    """Split docs into overlapping chunks for better retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # tries these in order
    )
    chunks = splitter.split_documents(docs)
    print(f"[2/4] Split into {len(chunks)} chunks  (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks

# ── Step 3 & 4: Embed + Store ────────────────────────────────────────────────

def build_vector_store(chunks):
    """Embed chunks and persist them in ChromaDB."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[3/4] Embedded and stored {len(chunks)} chunks in ChromaDB → '{CHROMA_DIR}'")
    return vector_store

def load_vector_store():
    """Load an existing ChromaDB store (skip re-embedding)."""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    count = vector_store._collection.count()
    print(f"[3/4] Loaded existing ChromaDB with {count} chunks from '{CHROMA_DIR}'")
    return vector_store

# ── Step 5: Query ─────────────────────────────────────────────────────────────

def build_qa_chain(vector_store):
    """Build a RetrievalQA chain: retriever → LLM."""
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",          # "stuff" = inject all chunks into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": RAG_PROMPT},
    )
    print(f"[4/4] QA chain ready  (model={CHAT_MODEL}, top_k={TOP_K})\n")
    return qa_chain

def ask(qa_chain, question: str):
    """Run a question through the RAG pipeline and print the answer + sources."""
    print(f"Q: {question}")
    result = qa_chain.invoke({"query": question})

    print(f"\nA: {result['result']}")
    print("\n--- Sources ---")
    for i, doc in enumerate(result["source_documents"], 1):
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "?")
        print(f"  [{i}] {Path(source).name}, page {page + 1}")
        print(f"      ...{doc.page_content[:150].strip()}...")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # If ChromaDB already exists, skip re-indexing (saves time + API cost)
    if Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir()):
        vector_store = load_vector_store()
    else:
        docs         = load_documents()
        chunks       = split_documents(docs)
        vector_store = build_vector_store(chunks)

    qa_chain = build_qa_chain(vector_store)

    # ── Interactive loop ──
    print("Ask questions about your PDFs. Type 'quit' to exit.\n")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue
        ask(qa_chain, question)


if __name__ == "__main__":
    main()
