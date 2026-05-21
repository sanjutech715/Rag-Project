# =============================================================================
# Assignment 4: Build a Multi-Source RAG System
# Course      : Week 5 AI Class – RAG Assignments
# Institute   : Vidaamuyarchi Tech
# Submitted to: info@vidaamuyarchi.com
# =============================================================================

# ----------------------------- DEPENDENCIES ----------------------------------
# pip install langchain langchain-community faiss-cpu pypdf sentence-transformers
# pip install pandas beautifulsoup4 requests
# -----------------------------------------------------------------------------

import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

# ========================= SOURCE 1: PDF =====================================
def load_from_pdf(pdf_path: str):
    """Load documents from a PDF file."""
    print(f"[SOURCE 1] Loading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"  [WARN] PDF not found: {pdf_path}")
        return []
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "PDF"
    print(f"  → {len(docs)} pages loaded from PDF.")
    return docs

# ========================= SOURCE 2: CSV =====================================
def load_from_csv(csv_path: str):
    """Load documents from a CSV file."""
    print(f"[SOURCE 2] Loading CSV: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"  [WARN] CSV not found: {csv_path}")
        return []
    loader = CSVLoader(file_path=csv_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_type"] = "CSV"
    print(f"  → {len(docs)} rows loaded from CSV.")
    return docs

# ========================= SOURCE 3: WEBSITE =================================
def load_from_website(url: str):
    """Scrape and load content from a website."""
    print(f"[SOURCE 3] Loading Website: {url}")
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract visible text
        text = " ".join(
            tag.get_text(separator=" ", strip=True)
            for tag in soup.find_all(["p", "h1", "h2", "h3", "li"])
        )
        if not text.strip():
            print("  [WARN] No content extracted from website.")
            return []
        doc = Document(
            page_content=text,
            metadata={"source": url, "source_type": "Website"}
        )
        print(f"  → Website content loaded ({len(text)} characters).")
        return [doc]
    except Exception as e:
        print(f"  [ERROR] Failed to load website: {e}")
        return []

# ====================== COMBINE ALL SOURCES ==================================
def load_all_sources(pdf_path, csv_path, website_url):
    """Merge documents from all three sources."""
    all_docs = []
    all_docs.extend(load_from_pdf(pdf_path))
    all_docs.extend(load_from_csv(csv_path))
    all_docs.extend(load_from_website(website_url))
    print(f"\n[INFO] Total documents from all sources: {len(all_docs)}")
    return all_docs

# ====================== BUILD RETRIEVAL PIPELINE =============================
def build_multi_source_pipeline(all_docs):
    """Chunk, embed, and build FAISS vector store."""
    print("[INFO] Splitting and embedding documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)
    print(f"[INFO] Total chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("[INFO] Multi-source vector store ready.")
    return vector_store

# ========================= QUERY THE SYSTEM ==================================
def query_multi_source(vector_store, query: str, top_k=4):
    """Retrieve answers from multi-source pipeline."""
    print(f"\n[QUERY] {query}")
    results = vector_store.similarity_search(query, k=top_k)

    print("\n[Retrieved Chunks]")
    for i, doc in enumerate(results, 1):
        src_type = doc.metadata.get("source_type", "Unknown")
        src      = doc.metadata.get("source", doc.metadata.get("source", ""))
        print(f"  [{i}] Source: {src_type} ({src})")
        print(f"       Content: {doc.page_content[:150].strip()}...")

    # Combine context and generate answer using LLM
    context = "\n\n".join([doc.page_content for doc in results])
    llm = Ollama(model="mistral")
    prompt = f"""Answer the question based on the context below.
Context:
{context}

Question: {query}
Answer:"""
    answer = llm.invoke(prompt)
    print(f"\n[Answer] {answer}")
    return answer

# ================================ MAIN =======================================
if __name__ == "__main__":
    # ----- Configure your sources -----
    PDF_PATH    = "sample_document.pdf"        # Your PDF file
    CSV_PATH    = "sample_data.csv"            # Your CSV file
    WEBSITE_URL = "https://en.wikipedia.org/wiki/Artificial_intelligence"

    all_docs     = load_all_sources(PDF_PATH, CSV_PATH, WEBSITE_URL)
    vector_store = build_multi_source_pipeline(all_docs)

    # ----- Sample Queries -----
    TEST_QUERIES = [
        "What is artificial intelligence?",
        "List the main topics covered in the document.",
        "What data is available in the CSV?"
    ]

    for q in TEST_QUERIES:
        query_multi_source(vector_store, q)

# ========================= SAMPLE OUTPUT =====================================
# [QUERY] What is artificial intelligence?
# [Retrieved Chunks]
#   [1] Source: Website (https://en.wikipedia.org/...)
#       Content: Artificial intelligence (AI) is intelligence...
#   [2] Source: PDF (sample_document.pdf)
#       Content: AI systems are designed to...
# [Answer] Artificial intelligence refers to...
# =============================================================================
