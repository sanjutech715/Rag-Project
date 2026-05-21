# Assignment 04 – Multi-Source RAG System

**Course:** Week 5 AI Class – RAG Assignments
**Institute:** Vidaamuyarchi Tech
**Contact:** info@vidaamuyarchi.com

---

## 📌 Overview

A standard RAG system pulls from a single source (e.g., one PDF). This project builds a **Multi-Source RAG** that combines knowledge from:
- 📄 **PDF** documents
- 📊 **CSV** files (structured data)
- 🌐 **Websites** (live web scraping)

All sources are merged into a single FAISS vector store and queried together.

---

## 🧠 How It Works

```
Source 1: PDF ──┐
Source 2: CSV ──┼──→ Load & Parse ──→ Merge All Docs
Source 3: URL ──┘
                        ↓
               Chunk with RecursiveTextSplitter
                        ↓
               HuggingFace Embeddings
                        ↓
               FAISS Vector Store (unified)
                        ↓
               Query → Retrieve → Ollama LLM → Answer
                        ↓
               Source attribution in response
```

---

## 📁 Project Structure

```
assignment_04_multi_source_rag/
├── multi_source_rag.py    # Multi-source RAG script
├── sample_data.csv        # Example CSV data (add your own)
├── .env.example           # Environment variable template
├── requirements.txt       # Python dependencies
├── .gitignore             # Git ignore rules
└── README.md              # Project documentation
```

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/YOUR_USERNAME/assignment_04_multi_source_rag.git
cd assignment_04_multi_source_rag

python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows

pip install -r requirements.txt

# Pull Ollama model
ollama pull mistral
```

---

## ▶️ How to Run

```bash
python multi_source_rag.py
```

Update the paths/URLs in the script:
```python
PDF_PATH  = "your_document.pdf"
CSV_PATH  = "your_data.csv"
WEBSITE   = "https://example.com"
```

**Sample Output:**
```
[SOURCE 1] Loading PDF: document.pdf → 12 pages loaded
[SOURCE 2] Loading CSV: data.csv → 50 rows loaded
[SOURCE 3] Loading Website: https://example.com → 1 page scraped

Total documents merged: 63
FAISS vector store created.

Q: What does the CSV data show?
A: Based on the CSV source, the data shows... [Source: CSV]
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Loader | LangChain PyPDFLoader |
| CSV Loader | LangChain CSVLoader |
| Web Scraper | BeautifulSoup4 + Requests |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM | Ollama (Mistral) |

---

## 📚 References

- [LangChain Document Loaders](https://python.langchain.com/docs/modules/data_connection/document_loaders/)
- [BeautifulSoup Docs](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Multi-Source RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
