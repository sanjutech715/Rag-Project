# Assignment 03 – Hybrid Search (BM25 + Vector Search)

**Course:** Week 5 AI Class – RAG Assignments
**Institute:** Vidaamuyarchi Tech
**Contact:** info@vidaamuyarchi.com

---

## 📌 Overview

Standard RAG uses only **semantic vector search** — which can miss exact keyword matches. **Hybrid Search** combines:
- 🔢 **BM25** (keyword-based, TF-IDF style) — great for exact terms
- 🧠 **Vector Search** (semantic, FAISS) — great for meaning & context
- ⚖️ **Combined Score** — best of both worlds

---

## 🔍 Search Methods Compared

| Method | Technique | Strength |
|--------|-----------|----------|
| Vector Search | Cosine similarity on embeddings | Semantic understanding |
| BM25 Search | Term frequency / keyword matching | Exact word match |
| **Hybrid Search** | Weighted combination of both | Best overall accuracy |

---

## 🧠 How It Works

```
PDF → Load → Chunk
   ↓                    ↓
FAISS Vector Store    BM25 Index
   ↓                    ↓
Vector Scores  +  BM25 Scores
        ↓
  Weighted Fusion (α × vector + (1-α) × BM25)
        ↓
  Top-K Re-ranked Results
```

---

## 📁 Project Structure

```
assignment_03_hybrid_search/
├── hybrid_search.py     # Hybrid search implementation
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/YOUR_USERNAME/assignment_03_hybrid_search.git
cd assignment_03_hybrid_search

python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python hybrid_search.py
```

**Sample Output:**
```
====== Vector Search Results ======
1. [Score: 0.82] "The transformer architecture uses self-attention..."
2. [Score: 0.79] "Attention mechanisms allow the model to..."

====== BM25 Search Results ======
1. [Score: 4.21] "The transformer was introduced in 2017..."
2. [Score: 3.98] "Transformer models are used in NLP..."

====== Hybrid Search Results ======
1. [Combined: 0.91] "The transformer architecture uses self-attention..."
2. [Combined: 0.87] "Transformer models are used in NLP..."
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Keyword Search | BM25 (rank-bm25) |
| Semantic Search | FAISS + HuggingFace Embeddings |
| Score Fusion | Weighted linear combination |
| Framework | LangChain |

---

## 📚 References

- [BM25 Algorithm Explained](https://en.wikipedia.org/wiki/Okapi_BM25)
- [Hybrid Search in RAG – Pinecone](https://www.pinecone.io/learn/hybrid-search-intro/)
- [rank-bm25 Library](https://github.com/dorianbrown/rank_bm25)
