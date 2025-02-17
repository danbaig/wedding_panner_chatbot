# Wedding Planner Chatbot with Multi-Agent Flow (GPT-4)

This repository contains a **wedding planning chatbot** named “Sara.” It uses [Streamlit](https://streamlit.io/) for the front-end interface, and incorporates multiple tools, agents, and retrieval components (via [LangChain](https://github.com/hwchase17/langchain) and [LangGraph](https://github.com/langgraph-ai/langgraph)) to handle user queries about wedding services, budgeting, and vendors. The chatbot interacts in a conversational manner and uses vendor data stored in a **Chroma** vector store.

---

## Table of Contents

1. [Features](#features)  
2. [Installation & Setup](#installation--setup)  
3. [Usage](#usage)  
4. [Files and Structure](#files-and-structure)  
5. [Environment Variables](#environment-variables)  
6. [Data Sources](#data-sources)  
7. [Customization](#customization)  
8. [License](#license)  

---

## Features

- **Multi-step Onboarding**: “Sara” collects basic wedding details (name, spouse name, location, budget, date) before launching into open conversation.  
- **Budgeting Agent**: Dynamically allocates the user’s budget across different categories (like Apparel, Reception, Photography, etc.) or provides a detailed breakdown for a specific category.  
- **Vendor Retrieval**: Retrieves vendor information from a Chroma-based vector store, using semantic similarity.  
- **Conversational Memory**: The chatbot retains user context (via a conversation buffer) for more natural follow-up responses.  
- **Tool-Based Architecture**: The code uses multiple discrete “tools” for budget breakdown, vendor lookup, general Q&A, etc. Tools are orchestrated by a state machine (LangGraph).  
- **GPT-4 & OpenAI**: Uses OpenAI’s GPT-4 model for generating high-quality, context-aware responses.

---

## Installation & Setup

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/wedding-planner-chatbot.git
   cd wedding-planner-chatbot
2. python -m venv venv
source venv/bin/activate  # on Linux/Mac
# or
venv\Scripts\activate  # on Windows
3. pip install -r requirements.txt
4. OPENAI_API_KEY=sk-XXXXXXXXXXXX
5. streamlit run app.py
