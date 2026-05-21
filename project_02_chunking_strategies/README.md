# Assignment 02 – Chunking Strategies in RAG

**Course:** Week 5 AI Class – RAG Assignments
**Institute:** Vidaamuyarchi Tech
**Contact:** info@vidaamuyarchi.com

---

## 📌 Overview

Chunking is a critical step in RAG pipelines. How you split documents directly affects retrieval quality. This project compares **3 different chunking strategies** and benchmarks their performance.

---

## 🔪 Chunking Strategies Compared

| Strategy | Description | Chunk Size | Overlap |
|----------|-------------|-----------|---------|
| **Fixed-Size** | Split by character count, no overlap | 500 | 0 |
| **Sliding Window** | Fixed-size with overlap for context continuity | 500 | 100 |
| **Semantic** | Split by paragraphs/sections (natural boundaries) | Variable | Variable |

---

## 🧠 How It Works

```
PDF File
   ↓
Load (PyPDF)
   ↓
Apply Strategy 1: Fixed-Size Chunking
Apply Strategy 2: Sliding Window Chunking
Apply Strategy 3: Semantic Chunking
   ↓
Generate Embeddings for each strategy
   ↓
Build FAISS Vector Stores
   ↓
Run Same Query on All 3
   ↓
Compare Results (chunk count, retrieval quality, time)
```

---

## 📁 Project Structure

```
assignment_02_chunking_strategies/
├── chunking_strategies.py   # Chunking comparison script
├── .env.example             # Environment variable template
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # Project documentation
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/YOUR_USERNAME/assignment_02_chunking_strategies.git
cd assignment_02_chunking_strategies

python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

---

## ▶️ How to Run

```bash
python chunking_strategies.py
```

Place your PDF file in the folder and update `PDF_PATH` in the script.

**Sample Output:**
```
[Strategy 1] Fixed-Size Chunking
  → Total chunks: 142

[Strategy 2] Sliding Window Chunking (with overlap)
  → Total chunks: 178

[Strategy 3] Semantic Chunking (Paragraph/Section Based)
  → Total chunks: 95

====== COMPARISON RESULTS ======
Strategy 1 | Chunks: 142 | Time: 1.2s
Strategy 2 | Chunks: 178 | Time: 1.5s
Strategy 3 | Chunks:  95 | Time: 0.9s
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Loader | LangChain PyPDFLoader |
| Chunking | CharacterTextSplitter, RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (CPU) |

---

## 📚 References

- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [Chunking Strategies for RAG – Blog](https://www.pinecone.io/learn/chunking-strategies/)
