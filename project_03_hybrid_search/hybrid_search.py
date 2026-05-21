# =============================================================================
# Assignment 3: Implement Hybrid Search (BM25 + Vector Search)
# Course      : Week 5 AI Class – RAG Assignments
# Institute   : Vidaamuyarchi Tech
# Submitted to: info@vidaamuyarchi.com
# =============================================================================

# ----------------------------- DEPENDENCIES ----------------------------------
# pip install langchain langchain-community faiss-cpu pypdf sentence-transformers
# pip install rank-bm25
# -----------------------------------------------------------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
import numpy as np

# ======================== LOAD & CHUNK PDF ====================================
def load_and_chunk(pdf_path: str):
    print(f"[INFO] Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"[INFO] Total chunks: {len(chunks)}")
    return chunks

# =================== BUILD VECTOR STORE (Semantic Search) ====================
def build_vector_store(chunks):
    print("[INFO] Building FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

# =================== BUILD BM25 INDEX (Keyword Search) =======================
def build_bm25_index(chunks):
    print("[INFO] Building BM25 keyword index...")
    tokenized_corpus = [
        chunk.page_content.lower().split() for chunk in chunks
    ]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25

# =================== VECTOR-ONLY SEARCH ======================================
def vector_search(vector_store, query: str, top_k=5):
    """Retrieve documents using semantic vector similarity."""
    results = vector_store.similarity_search_with_score(query, k=top_k)
    return [(doc.page_content, score) for doc, score in results]

# =================== BM25-ONLY SEARCH ========================================
def bm25_search(bm25, chunks, query: str, top_k=5):
    """Retrieve documents using BM25 keyword search."""
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i].page_content, scores[i]) for i in top_indices]

# =================== HYBRID SEARCH (BM25 + Vector) ===========================
def hybrid_search(bm25, chunks, vector_store, query: str, top_k=5, alpha=0.5):
    """
    Combine BM25 and vector search scores.
    alpha=0.5 means equal weight to both methods.
    Increase alpha → more weight to vector search.
    Decrease alpha → more weight to BM25.
    """
    print(f"\n[Hybrid Search] Query: {query}")

    # --- BM25 Scores ---
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    # Normalize BM25 scores to [0, 1]
    bm25_max = max(bm25_scores) if max(bm25_scores) > 0 else 1
    bm25_norm = bm25_scores / bm25_max

    # --- Vector Scores ---
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    query_embed = embeddings.embed_query(query)
    vector_results = vector_store.similarity_search_with_score(query, k=len(chunks))
    vector_score_map = {doc.page_content: score for doc, score in vector_results}

    # --- Combine Scores ---
    combined = []
    for i, chunk in enumerate(chunks):
        vec_score = vector_score_map.get(chunk.page_content, 1.0)
        # FAISS returns L2 distance; convert to similarity
        vec_sim = 1 / (1 + vec_score)
        hybrid_score = alpha * vec_sim + (1 - alpha) * bm25_norm[i]
        combined.append((chunk.page_content, hybrid_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]

# ========================= COMPARE METHODS ====================================
def compare_search_methods(pdf_path: str, queries: list):
    chunks       = load_and_chunk(pdf_path)
    vector_store = build_vector_store(chunks)
    bm25         = build_bm25_index(chunks)

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        print("\n[1] Vector-Only Search:")
        for i, (text, score) in enumerate(vector_search(vector_store, query, top_k=3), 1):
            print(f"  [{i}] Score: {score:.4f} | {text[:120].strip()}...")

        print("\n[2] BM25-Only Search:")
        for i, (text, score) in enumerate(bm25_search(bm25, chunks, query, top_k=3), 1):
            print(f"  [{i}] Score: {score:.4f} | {text[:120].strip()}...")

        print("\n[3] Hybrid Search (BM25 + Vector, alpha=0.5):")
        for i, (text, score) in enumerate(hybrid_search(bm25, chunks, vector_store, query, top_k=3), 1):
            print(f"  [{i}] Score: {score:.4f} | {text[:120].strip()}...")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print("Hybrid Search combines keyword precision (BM25) with")
    print("semantic understanding (Vector Search), producing more")
    print("accurate and relevant results than either method alone.")

# ================================ MAIN =======================================
if __name__ == "__main__":
    PDF_PATH = "sample_document.pdf"   # Replace with your PDF path

    TEST_QUERIES = [
        "What is the main objective?",
        "Explain the methodology used.",
        "What are the final results?"
    ]

    compare_search_methods(PDF_PATH, TEST_QUERIES)
# =============================================================================
