# 🚀 Building a Multi-Document RAG System

## 📄 Overview

This project demonstrates how to build a **Multi-Document Retrieval-Augmented Generation (RAG) system** that enables intelligent querying across multiple PDF documents.

The system processes a collection of documents, retrieves the most relevant information, and generates accurate answers using a local Large Language Model (LLM).

It is designed to simulate real-world applications such as:

* 📚 Research assistants
* 📊 Document analysis systems
* 💬 Intelligent chatbots

---

## 🧠 Key Features

* 📂 Multi-document ingestion (PDF support)
* ✂️ Smart text chunking for better retrieval
* 🔎 Semantic search using embeddings
* 🧬 Vector storage with ChromaDB
* 🤖 Local LLM integration (Ollama – LLaMA 3)
* 💬 Interactive query system
* 🔐 Fully offline and cost-efficient

---

## 🏗️ System Workflow

```id="flow001"
Multiple PDFs → Chunking → Embeddings → Vector Database
                                              ↓
User Query → Retriever → Relevant Context → LLM → Answer
```

---

## ⚙️ Tech Stack

* **Python**
* **LangChain**
* **ChromaDB**
* **HuggingFace Embeddings**
* **Ollama (LLaMA 3)**

---

## 📦 Installation

```bash id="inst001"
pip install -r requirements.txt
```

---

## ⚡ Setup

```bash id="setup101"
ollama pull llama3
ollama serve
```

---

## ▶️ Running the Project

```bash id="run001"
python main.py
```

---

## 💡 Example Query

```id="ex001"
"What are the main concepts explained across the documents?"
```

---

## 🎯 Use Cases

* Multi-document summarization
* Knowledge base assistants
* Academic research support
* AI-powered document search

---

## 🔥 Future Improvements

* Hybrid search (semantic + keyword)
* Web UI using Streamlit
* Multi-turn conversation support
* Metadata filtering

---

## 📬 Contact

**ABHINASH BEVARA**

Email: [abhibevera06@gmail.com](mailto:your-email@example.com)
