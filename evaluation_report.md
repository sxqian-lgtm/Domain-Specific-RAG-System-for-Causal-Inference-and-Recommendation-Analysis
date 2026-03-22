# Evaluation Report

## Setup

- Compared direct Gemini answers (`no_rag`) against retrieval-augmented answers (`rag`).
- Questions were drawn from the project evaluation set in `data/eval_questions/starter_questions.txt`.
- Scoring dimensions: relevance, grounding, reasoning quality.
- This report currently summarizes a partial run rather than the full 33-question evaluation set.
- Completed questions in this partial run: 5

## Average Scores

- No-RAG relevance: 5.0
- No-RAG grounding: 5.0
- No-RAG reasoning quality: 5.0
- RAG relevance: 5.0
- RAG grounding: 5.0
- RAG reasoning quality: 5.0

## Summary

The current pilot results show that both no-RAG and RAG answers score strongly on the completed questions. This is encouraging, but it should not yet be treated as a final comparison because the run covers only a subset of the question set.

RAG is still expected to be more valuable on grounding and project specificity because answers are tied to curated project chunks, especially for the Netflix-style case and synthetic DID result questions. No-RAG answers can still sound fluent, but they are more likely to be generic or less aligned with the exact project setup.

## Notes

- This is a first-pass partial evaluation intended for project development and README reporting.
- Scores come from an LLM-based judge, so they should be treated as approximate rather than final human labels.
- The remaining questions were not completed because Gemini free-tier quota was exhausted during evaluation.
- A smaller core evaluation set and a low-quota mode were added so the final evaluation can be resumed later.
