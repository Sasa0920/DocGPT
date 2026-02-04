# 📚 DocGPT – AI-Powered Document Chatbot

DocGPT is an AI-powered document assistant that allows users to upload PDF documents and ask questions about their content using natural language.  
It uses **Retrieval-Augmented Generation (RAG)** with local LLM inference powered by **Ollama**.

---

## 🚀 Features

- 📄 Upload and process **multiple PDF documents**
- 🔍 Intelligent text chunking and vector-based retrieval
- 🤖 Conversational Q&A using a **local LLM (Mistral)**
- 🧠 Maintains chat history for contextual conversations
- ⚡ Fully offline (no OpenAI API required)
- 🎨 Clean and interactive UI built with Streamlit

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit  
- **LLM**: Mistral (via Ollama)  
- **Framework**: LangChain  
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)  
- **Vector Store**: FAISS  
- **PDF Parsing**: PyPDF2  
- **Language**: Python  

---

## 🧠 Architecture (RAG Pipeline)

1. User uploads PDF files  
2. PDFs are converted to raw text  
3. Text is split into overlapping chunks  
4. Chunks are embedded using HuggingFace embeddings  
5. FAISS stores vectors for fast similarity search  
6. Relevant chunks are retrieved for each query  
7. Mistral LLM generates answers using retrieved context  

---
