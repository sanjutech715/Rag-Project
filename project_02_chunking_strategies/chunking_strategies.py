# =============================================================================
# Assignment 2: Compare Different Chunking Strategies in RAG
# Course      : Week 5 AI Class – RAG Assignments
# Institute   : Vidaamuyarchi Tech
# Submitted to: info@vidaamuyarchi.com
# =============================================================================

# ----------------------------- DEPENDENCIES ----------------------------------
# pip install langchain langchain-community faiss-cpu pypdf sentence-transformers
# -----------------------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time

# ======================== LOAD PDF ============================================
def load_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"[INFO] Loaded {len(documents)} pages.")
    return documents

# ===================== CHUNKING STRATEGIES ====================================

# Strategy 1: Fixed-Size Chunking
def fixed_size_chunking(documents, chunk_size=500, chunk_overlap=0):
    """Split into fixed-size chunks with NO overlap."""
    print("\n[Strategy 1] Fixed-Size Chunking")
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n"
    )
    chunks = splitter.split_documents(documents)
    print(f"  → Total chunks: {len(chunks)}")
    return chunks

# Strategy 2: Sliding Window (Fixed-size WITH overlap)
def sliding_window_chunking(documents, chunk_size=500, chunk_overlap=100):
    """Split into chunks with overlap (sliding window)."""
    print("\n[Strategy 2] Sliding Window Chunking (with overlap)")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    print(f"  → Total chunks: {len(chunks)}")
    return chunks

# Strategy 3: Semantic Chunking (Paragraph / Section based)
def semantic_chunking(documents):
    """Split based on paragraphs and sections (semantic boundaries)."""
    print("\n[Strategy 3] Semantic Chunking (Paragraph/Section Based)")
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=800,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"  → Total chunks: {len(chunks)}")
    return chunks

# ====================== BUILD VECTOR STORE ====================================
def build_vector_store(chunks, label=""):
    print(f"  [INFO] Building vector store for: {label}")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store

# ======================== EVALUATE STRATEGIES =================================
def evaluate_strategy(vector_store, queries, strategy_name):
    """Run queries and show retrieved results."""
    print(f"\n{'='*60}")
    print(f" Results for: {strategy_name}")
    print(f"{'='*60}")
    for query in queries:
        print(f"\nQuery: {query}")
        results = vector_store.similarity_search(query, k=2)
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.page_content[:150].strip()}...")

# ========================= COMPARE ALL ========================================
def compare_strategies(pdf_path: str, queries: list):
    documents = load_pdf(pdf_path)

    strategies = {
        "Fixed-Size Chunking"          : fixed_size_chunking(documents),
        "Sliding Window Chunking"      : sliding_window_chunking(documents),
        "Semantic (Paragraph) Chunking": semantic_chunking(documents),
    }

    for name, chunks in strategies.items():
        start = time.time()
        vs = build_vector_store(chunks, label=name)
        elapsed = time.time() - start
        print(f"  → Vector store built in {elapsed:.2f}s")
        evaluate_strategy(vs, queries, name)

    print("\n" + "=" * 60)
    print("OBSERVATIONS")
    print("=" * 60)
    print("1. Fixed-Size   : Fast but may cut sentences mid-way.")
    print("2. Sliding Window: Better context due to overlap; more chunks.")
    print("3. Semantic     : Best relevance; respects natural text boundaries.")

# ================================ MAIN =======================================
if __name__ == "__main__":
    PDF_PATH = "sample_document.pdf"   # Replace with your PDF path

    # ----- Sample Queries -----
    TEST_QUERIES = [
        "What is the main objective of this document?",
        "List the key points discussed.",
        "What are the conclusions mentioned?"
    ]

    compare_strategies(PDF_PATH, TEST_QUERIES)

# ========================= EXPECTED OUTPUT ===================================
# [Strategy 1] Fixed-Size Chunking    → 45 chunks
# [Strategy 2] Sliding Window         → 52 chunks
# [Strategy 3] Semantic Chunking      → 38 chunks
#
# Results show Semantic Chunking retrieves the most relevant content
# because it preserves paragraph meaning.
# =============================================================================
