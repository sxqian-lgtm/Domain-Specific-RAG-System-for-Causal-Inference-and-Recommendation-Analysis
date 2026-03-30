# Evaluation Report

## Setup

- Compared direct Gemini answers (`no_rag`) against retrieval-augmented answers (`rag`).
- Questions were drawn from a project evaluation set in `data/eval_questions/`.
- Scoring dimensions: relevance, grounding, reasoning quality.
- Questions evaluated so far: 13
- Scored answer rows so far: 26

## Average Scores

- No-RAG relevance: 5.0
- No-RAG grounding: 4.92
- No-RAG reasoning quality: 5.0
- RAG relevance: 5.0
- RAG grounding: 5.0
- RAG reasoning quality: 5.0

## Summary

RAG is expected to perform better on grounding because answers are tied to curated project chunks, especially for the Netflix-style case and synthetic DID result questions. No-RAG answers can still sound fluent, but they are more likely to be generic or less aligned with the exact project setup.

## Notes

- This is a first-pass evaluation intended for project development and README reporting.
- Scores come from an LLM-based judge, so they should be treated as approximate rather than final human labels.
