# Domain-Specific RAG System for Causal Inference and Recommendation Analysis

This project is based on the open-source LangChain RAG pipeline and adapts it into a domain-specific assistant for causal inference explanation and recommendation experiment analysis.

## Overview

The goal of this project is to build a local retrieval-augmented generation system that can:

- explain causal inference concepts such as DID, IPW, ATT, ATE, backdoor adjustment, and counterfactuals
- answer project-specific questions about a Netflix-style recommendation analysis case
- retrieve grounded evidence from curated project notes and method summaries
- support a broader causal AI workflow for analysis, explanation, and technical communication

The system uses Sentence Transformers for embeddings, FAISS for local retrieval, and Gemini for answer generation.

## Method

The current pipeline follows this flow:

1. Load local public-safe text chunks from the project knowledge base
2. Split documents into retrieval chunks
3. Build a FAISS vector index using `all-MiniLM-L6-v2`
4. Route each question into one of three modes:
   - `method`
   - `project`
   - `mixed`
5. Retrieve top-k relevant chunks
6. Format the retrieved context
7. Generate a grounded answer with Gemini through `google.genai`

This means the answer flow is:

`query -> route -> retrieve -> context -> Gemini -> answer`

## Data

The knowledge base is organized into:

- `data/causal_methods/`
- `data/netflix_did/`
- `data/project_notes/`
- `data/eval_questions/`

Private source materials are stored locally under:

- `data/raw_private/`

These raw materials are not intended for public redistribution. The public knowledge base is built from author-written summaries and chunked notes derived from those materials.

Current source categories include:

- causal inference textbook material and solution notes
- Netflix-style recommendation analysis project slides
- synthetic recommendation/viewing dataset
- project-specific analysis notes

## Project Structure

```text
rag-from-scratch/
├── app/
│   └── streamlit_app.py
├── artifacts/
│   └── faiss_index/
├── data/
│   ├── causal_methods/
│   ├── netflix_did/
│   ├── project_notes/
│   ├── eval_questions/
│   └── raw_private/
├── src/
│   ├── load_data.py
│   ├── split_docs.py
│   ├── build_index.py
│   ├── retrieve.py
│   ├── router.py
│   ├── prompt.py
│   ├── llm_gemini.py
│   ├── answer.py
│   └── evaluate.py
├── evaluation_results.md
├── evaluation_report.md
├── plan.txt
├── project_status.txt
└── requirements.txt
```

## Your Contribution

Compared with the original open-source baseline, the main project-specific contributions are:

- reorganized the notebook-style base into a script-based project structure
- replaced the generic demo knowledge source with a domain-specific causal inference and recommendation knowledge base
- switched the generation layer to Gemini through `google.genai`
- replaced the vector store setup with a local FAISS pipeline using Sentence Transformers
- added rule-based query routing for `method`, `project`, and `mixed` questions
- added a project-specific evaluation workflow and reporting files
- created public-safe text chunks derived from local source materials

## Evaluation

An evaluation pipeline is implemented to compare:

- `no_rag`
- `rag`

Scoring dimensions:

- relevance
- grounding
- reasoning quality

Current status:

- the evaluation script and question set are complete
- a pilot evaluation run has been completed
- the full 19-question run was interrupted by the Gemini free-tier daily quota

See:

- `evaluation_results.md`
- `evaluation_report.md`
- `project_status.txt`

## Demo

Example supported questions:

- Why can naive DID be biased
- Explain ATT versus ATE
- What does d separation mean in a causal graph
- What is the DID estimate in the synthetic Netflix style data
- How does the Netflix project connect business questions to causal inference

The Streamlit app is available in:

- `app/streamlit_app.py`

## Setup

Create and activate the Conda environment:

```powershell
conda activate RAG_S
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Build the index:

```powershell
python -m src.build_index
```

Ask a question:

```powershell
python -m src.answer "Why can naive DID be biased?"
```

Run the Streamlit app:

```powershell
streamlit run app/streamlit_app.py
```

## Notes

- This repository is based on the open-source LangChain RAG pipeline.
- Raw private files are intentionally separated from public-safe retrieval text.
- API keys should remain in `.env` and should never be committed to source control.
- The full evaluation should be rerun when Gemini quota is available again.

