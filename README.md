# AskMyDoc-AI
AskMyDocs is an AI-powered chatbot that lets you upload PDFs and ask context-aware questions with chat history support
# 📄 AskMyDocs — Chat with Your PDFs

> 🤖 An AI-powered RAG chatbot that lets you upload PDFs and have context-aware conversations with full chat history support.

---

## 🚀 Features

- 📂 Upload and parse PDF documents
- 🧠 Retrieval-Augmented Generation (RAG) using LangChain + ChromaDB
- 💬 Context-aware Q&A with memory-powered chat
- 🗃️ Session-based chat history
- 🔐 OpenAI and HuggingFace model integration

---

## 🛠️ Tech Stack

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| Streamlit       | Web UI for file upload and chat  |
| LangChain       | RAG pipeline & prompt chaining   |
| ChromaDB        | Vector store for document chunks |
| HuggingFace     | Embedding model for vectorization |
| OpenAI / Groq   | LLM backend for response generation |
| PyPDF / LangChain Loader | PDF parsing and document extraction |

---

## 🖼️ UI Preview

![screenshot](https://via.placeholder.com/900x400?text=UI+Preview+Here)

---

## 🔧 Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/your-username/askmydocs.git
cd askmydocs

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your environment variables
# Create a .env file and add:
# Hugging_face_token=your_token_here

# 5. Run the app
streamlit run app.py
