# Ell-ena RAG Context Optimization Prototype

This is a proof-of-concept pipeline built for the **AOSSIE GSoC 2026 Ell-ena** project. 

### The Core Problem
To migrate Ell-ena to **self-hosted open-source language models**, the system must account for significantly smaller context windows. Passing an entire database of past Kanban tickets into a local LLM prompt will cause token overflows and hallucinations.

### The Solution
This prototype demonstrates a strict extraction and filtering pipeline designed to reduce token overhead by ~90% before hitting the LLM API. 

1. **Context Filtering:** It uses TF-IDF and Cosine Similarity to compare a raw meeting transcript against existing open tickets, fetching only the top 2 most relevant past tickets.
2. **Strict Prompting:** It passes only the transcript and the filtered context to the LLM. 
3. **Validated Extraction:** It forces the LLM to ignore existing tasks and extract only *new* action items, outputting a strict JSON schema (`Assignee`, `Deadline`, `Task Title`, `Priority`) ready to be pushed to a Kanban backend.

4. ### Proof of Work: Execution Pipeline
The local execution logs below demonstrate the end-to-end RAG pipeline successfully ingesting data, scoring similarities, and formatting the final JSON output. *(Note: LLM generation was routed to a mock fallback to validate the retrieval architecture and prompt formatting without burning API credits).*

<img width="537" height="881" alt="Local Execution Terminal Logs" src="https://github.com/user-attachments/assets/47179914-e4d7-4ad9-85c1-b38ce28d208d" />


### Running Locally
1. Clone the repository.
2. Install dependencies: `pip install scikit-learn openai`
3. Add your LLM API key to `main.py` (or let it run the mock fallback to see the architecture).
4. Run: `python main.py`
