# 🎓 NLP & LLM Course Assistant (RAG System)

Project developed by **Group 1 – University of Salerno (UNISA)**  
Course: *Natural Language Processing & Large Language Models*

This repository hosts a **Conversational AI Assistant** based on **RAG (Retrieval-Augmented Generation)** architecture.  
The system is designed to answer *vertical questions* about the NLP course, leveraging **local documents** and **LLM models running on-premise via Ollama**.

---

## 🎯 Project Scope

The main objective is to provide a **highly specialized vertical assistant** capable of answering precisely using **exclusively the information contained in the course teaching materials**.

To overcome the limitations of traditional RAG systems, the project implements advanced strategies:

### 🔐 Contextualization & Prompt Engineering
- Guardrails and prompt engineering techniques
- Reduced hallucinations
- Strict domain constraint for reliable answers

### 🧠 “Manual” Memory Management
Instead of relying on implicit model memory, the system actively manages context through:

- **Automatic Detection**  
  Intelligent identification of follow-up questions

- **Query Rephrasing**  
  Dynamic reformulation of user queries enriched with semantic context from previous interactions  
  This improves retrieval accuracy when questions are ambiguous or incomplete

---

## 🌟 Key Features

### 🧠 Hybrid RAG Architecture (Custom)

Unlike standard RAG systems, this project implements an advanced **hybrid search logic** in `RAG.py`:

- **Vector Search**  
  Uses **FAISS** and `nomic-embed-text` embeddings for semantic similarity

- **Keyword Boosting**  
  Custom TF-IDF-like algorithm that:
  - Computes keyword importance in documents
  - Reduces vector distance when exact keyword matches occur

- **Reranking**  
  Final results are reordered by combining:
  - Vector similarity scores
  - Lexical keyword matches

---

### 💭 Memory & Follow-up Management

The `memory_handler.py` module makes conversations natural and coherent:

- **Automatic Follow-up Detection**  
  A dedicated LLM analyzes whether the user input is a follow-up question  
  *(e.g. “And how does it work?”)*

- **Query Rephrasing**  
  Automatically rewrites the question by injecting previous context before querying the vector database

---

### 🖥️ Run the system

To run the system use 

```bash
streamlit run main_gui.py
```
---

## 🛠️ System Requirements

The project is designed to run **locally** to ensure data privacy.

- **Ollama** (must be installed and running)
- **Python package dependancy** must be installed using the ***requirements.txt*** file

👉 Download Ollama from the official website.

---

## 📦 Required Models

Run the following commands to download the required models:

```bash
# LLM model for generation (chat)
ollama pull qwen2.5:7b-instruct

# Model for vector embeddings
ollama pull nomic-embed-text
```
You can also change the embeddings and the llm by changing the name of the models in the ***system/config.json*** file. Be sure to download the model using ollama before running the system.

## 👥 Project Members (Group 1)

*   [Angelo Molinario](https://github.com/amolinario3)
*   [Antonio Sessa](https://github.com/Antuke)
*   [Massimiliano Ranauro](https://github.com/MassimilianoRanauro)
*   [Pietro Martano](https://github.com/pietroemme)
