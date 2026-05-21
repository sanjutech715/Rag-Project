# Assignment 05 – RAG with Re-ranking & Performance Evaluation

**Course:** Week 5 AI Class – RAG Assignments
**Institute:** Vidaamuyarchi Tech
**Contact:** info@vidaamuyarchi.com

---

## 📌 Overview

Basic RAG retrieves top-K chunks by vector similarity — but similarity doesn't always equal relevance. **Re-ranking** adds a second-pass scoring model to re-order retrieved chunks before passing them to the LLM.

This project compares:
- ❌ **RAG without Re-ranking** — simple top-K retrieval
- ✅ **RAG with Re-ranking** — BGE Reranker filters & reorders chunks

---

## 🧠 How Re-ranking Works

```
Query
  ↓
FAISS Vector Search → Top-10 Chunks (first pass)
  ↓
BGE Reranker (cross-encoder) → Re-scores each chunk
  ↓
Top-3 Most Relevant Chunks (second pass)
  ↓
LLM (Ollama/Mistral) → Final Answer
```

---

## 📊 Without vs With Re-ranking

| Feature | Without Re-ranking | With Re-ranking |
|---------|-------------------|----------------|
| Retrieval | Vector similarity only | Vector → Cross-encoder rescore |
| Accuracy | Moderate | Higher |
| Speed | Faster | Slightly slower |
| Context quality | May include noise | More precise |

---

## 📁 Project Structure

```
assignment_05_reranking/
├── reranking_rag.py     # Re-ranking RAG script
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/YOUR_USERNAME/assignment_05_reranking.git
cd assignment_05_reranking

python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

pip install -r requirements.txt

# Pull Ollama model
ollama pull mistral
```

> 💡 GPU recommended for BGE Reranker. First run downloads the reranker model (~1GB).

---

## ▶️ How to Run

```bash
python reranking_rag.py
```

**Sample Output:**
```
====== WITHOUT RE-RANKING ======
Answer: The document mentions transformers in several contexts...
Time: 2.3s

====== WITH RE-RANKING ======
Top chunks after reranking:
  [Score: 0.94] "Transformers use self-attention to..."
  [Score: 0.88] "The encoder-decoder architecture..."
Answer: Transformers use self-attention mechanisms to process...
Time: 3.1s

====== EVALUATION ======
Re-ranking improved relevance of top chunks significantly.
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Re-ranker | BGE Reranker (`BAAI/bge-reranker-base`) |
| LLM | Ollama (Mistral) |
| Framework | LangChain + FlagEmbedding |

---

## 📚 References

- [BGE Reranker (BAAI)](https://huggingface.co/BAAI/bge-reranker-base)
- [FlagEmbedding Library](https://github.com/FlagOpen/FlagEmbedding)
- [Re-ranking in RAG – Blog](https://www.pinecone.io/learn/series/rag/rerankers/)
- [LangChain Retriever Docs](https://python.langchain.com/docs/modules/data_connection/retrievers/)
