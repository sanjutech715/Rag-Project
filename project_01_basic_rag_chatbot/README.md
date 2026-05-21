# Assignment 01 – Basic RAG Chatbot for PDF Documents

**Course:** Week 5 AI Class – RAG Assignments
**Institute:** Vidaamuyarchi Tech
**Contact:** info@vidaamuyarchi.com

---

## 📌 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI technique that combines:
1. **Retrieval** – Finding relevant chunks from a document using vector search
2. **Generation** – Using an LLM to answer questions based on retrieved context

This project builds a simple RAG chatbot that answers questions from any PDF document.

---

## 🧠 How It Works

```
PDF File
   ↓
Load & Parse (PyPDF)
   ↓
Split into Chunks (RecursiveCharacterTextSplitter)
   ↓
Generate Embeddings (HuggingFace all-MiniLM-L6-v2)
   ↓
Store in FAISS Vector DB
   ↓
User Query → Retrieve Top-3 Chunks → LLM (Ollama/Mistral) → Answer
```

---

## 📁 Project Structure

```
assignment_01_basic_rag_chatbot/
├── rag_chatbot.py       # Main RAG chatbot script
├── .env.example         # Environment variable template
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- [Ollama](https://ollama.ai/) installed locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/assignment_01_basic_rag_chatbot.git
cd assignment_01_basic_rag_chatbot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env with your settings

# 5. Pull Ollama model
ollama pull mistral
```

---

## ▶️ How to Run

```bash
# Place your PDF in this folder, then:
python rag_chatbot.py
```

The chatbot will:
1. Load your PDF
2. Split into chunks & generate embeddings
3. Start an interactive Q&A session

**Sample Interaction:**
```
You: What is the main topic of this document?
Bot: The document discusses... [Page 1: ...]

You: exit
Goodbye!
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Loader | LangChain PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (CPU) |
| LLM | Ollama (Mistral) |
| Framework | LangChain |

---

## 📚 References

- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering/)
- [FAISS Vector Store](https://faiss.ai/)
- [Ollama – Local LLMs](https://ollama.ai/)
- [Sentence Transformers](https://www.sbert.net/)
