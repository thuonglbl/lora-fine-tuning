# Project: Corporate IT Chatbot

## Project Overview
This project is an internal AI assistant for employees, designed to answer IT-related questions and automate simple tasks. The entire system is built to be deployed on-premise to ensure data privacy and security.

## Core Goal
The primary goal is to provide fast and accurate answers to IT support questions by retrieving information from a private Confluence Knowledge Base. A secondary goal is to integrate with JIRA to provide automated, first-level support on newly created tickets.

## Architecture & Technology Stack
The system follows a multi-channel RAG (Retrieval-Augmented Generation) architecture.

- **Language:** Python 3.13+
- **AI Framework:** LangChain, with LangGraph for complex agentic orchestration.
- **LLM Serving:** Ollama (development) with plans to migrate to vLLM (production).
- **Generation LLM:** A fine-tuned version of `LLaMA 3.1 8B` using LoRA.
- **Embedding Model:** `all-MiniLM-L6-v2`.
- **Vector Database:** FAISS (in-memory) with plans to migrate to Qdrant (production).
- **Reranker:** FlashRank.
- **Web UI:** Streamlit.
- **Workflow Automation:** n8n (for JIRA integration).

## Key Architectural Concepts

- **Multi-Channel System:** The chatbot serves two distinct channels:
    1.  A real-time web UI built with **Streamlit**.
    2.  An automated workflow integrated with **JIRA** via **n8n**.

- **Central RAG Service:** The core RAG pipeline is a central Python/LangChain service. Both the Streamlit backend and the n8n workflow call this same service via an API. The service is stateless from the caller's perspective.

- **Model Context Protocol (MCP):** A specific, internally defined data structure is used to format all context (the user query, chat history, and retrieved document chunks) before it is sent to the LLM. This ensures consistency.

- **Channel-Specific Output Formatting:** The RAG pipeline itself is responsible for formatting the final answer. It checks the `source` of the query (`'streamlit'` or `'jira'`) and formats the LLM's base response appropriately for the target channel before returning it.

## Instructions for the AI Assistant

- **Your Role:** Act as an expert Python developer and MLOps engineer assisting with this project.
- **Code Style:** All Python code should be clean, modular, and follow PEP 8 standards. Include type hints where appropriate.
- **Technology Priority:** When generating code or suggesting solutions, prioritize using libraries and frameworks already in the technology stack (LangChain, LangGraph, Streamlit).
- **Architectural Awareness:** When answering questions about the system, refer to the multi-channel and central-service architecture described above.
- **On-Premise Focus:** Always remember that the entire stack is designed to be **on-premise**. Avoid suggesting solutions that rely on cloud-based, third-party APIs for core functionality unless explicitly asked.
- **LangGraph Pattern:** For new agentic features or complex logic flows, the preferred pattern is to use LangGraph. This involves adding new nodes for specific tasks and using conditional edges for routing.