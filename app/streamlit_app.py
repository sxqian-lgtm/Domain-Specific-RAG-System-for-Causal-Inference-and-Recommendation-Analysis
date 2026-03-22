"""Streamlit UI for the domain-specific RAG system."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.answer import answer_question


EXAMPLE_QUESTIONS = [
    "Why can naive DID be biased?",
    "Explain ATT versus ATE.",
    "What does d separation mean in a causal graph?",
    "What is the DID estimate in the synthetic Netflix style data?",
    "How does the Netflix project connect business questions to causal inference?",
]

ROUTE_LABELS = {
    "method": "Method",
    "project": "Project",
    "mixed": "Mixed",
}


def render_chunk_card(index: int, doc) -> None:
    category = doc.metadata.get("category", "unknown")
    filename = doc.metadata.get("filename", "unknown")
    with st.expander(f"Chunk {index} | {category} | {filename}", expanded=index == 1):
        st.caption(f"Category: {category}")
        st.write(doc.page_content)


st.set_page_config(page_title="Causal RAG Demo", layout="wide")

st.title("Causal Inference RAG Demo")
st.caption(
    "A domain-specific assistant for causal inference explanation and recommendation analysis, "
    "based on the open-source LangChain RAG pipeline."
)

with st.sidebar:
    st.header("Run Settings")
    top_k = st.slider("Top-k retrieved chunks", min_value=2, max_value=8, value=4)
    llm_model = st.selectbox("Gemini model", ["gemini-3-flash-preview"], index=0)
    st.markdown("---")
    st.subheader("Knowledge Sources")
    st.write("`causal_methods`")
    st.write("`netflix_did`")
    st.write("`project_notes`")
    st.markdown("---")
    st.subheader("Example Questions")
    for idx, sample in enumerate(EXAMPLE_QUESTIONS):
        if st.button(sample, key=f"example_{idx}", use_container_width=True):
            st.session_state["question_input"] = sample

if "question_input" not in st.session_state:
    st.session_state["question_input"] = EXAMPLE_QUESTIONS[0]

st.markdown(
    """
This demo routes each question into one of three retrieval modes:

- `method`: causal inference concepts and method explanations
- `project`: Netflix-style recommendation project notes and findings
- `mixed`: combined retrieval across both method and project knowledge
"""
)

question = st.text_area(
    "Ask a question",
    key="question_input",
    height=100,
    placeholder="Why can naive DID be biased?",
)

run_col, clear_col = st.columns([1, 1])
run_clicked = run_col.button("Generate Answer", type="primary", use_container_width=True)
clear_clicked = clear_col.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state["question_input"] = ""
    st.rerun()

if run_clicked:
    if not question.strip():
        st.warning("Enter a question before running the demo.")
    else:
        try:
            with st.spinner("Retrieving evidence and generating answer..."):
                answer, docs, route = answer_question(
                    question.strip(),
                    k=top_k,
                    llm_model=llm_model,
                )

            overview_col, route_col = st.columns([3, 1])
            with overview_col:
                st.subheader("Answer")
                st.write(answer)
            with route_col:
                st.subheader("Route")
                st.metric("Selected Mode", ROUTE_LABELS.get(route, route))
                st.metric("Chunks Returned", len(docs))

            st.subheader("Retrieved Evidence")
            if docs:
                for i, doc in enumerate(docs, start=1):
                    render_chunk_card(i, doc)
            else:
                st.info("No chunks were retrieved for this question.")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            st.info(
                "If the message mentions quota or resource exhaustion, the Gemini free tier "
                "may have reached its current limit."
            )
