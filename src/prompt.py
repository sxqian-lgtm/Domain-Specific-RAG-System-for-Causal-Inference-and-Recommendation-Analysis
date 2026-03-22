"""Prompt templates for grounded answer generation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


BASELINE_RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a careful AI assistant for causal inference and recommendation analysis.

Use only the retrieved context below to answer the user's question. If the context is
insufficient, say what is missing instead of inventing facts.

Retrieved context:
{context}

Question:
{question}

Answer with:
1. A direct answer
2. A short explanation grounded in the context
3. A brief note on uncertainty if the evidence is incomplete
"""
)


def render_baseline_prompt(context: str, question: str) -> str:
    return BASELINE_RAG_PROMPT.format(context=context, question=question)
