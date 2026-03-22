"""Evaluation utilities for comparing RAG and non-RAG answers."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.answer import answer_question
from src.llm_gemini import generate_with_gemini


QUESTION_FILE = "data/eval_questions/starter_questions.txt"
CORE_QUESTION_FILE = "data/eval_questions/core_eval_questions.txt"
RESULTS_JSONL = "artifacts/evaluation_results.jsonl"
RESULTS_MD = "evaluation_results.md"
REPORT_MD = "evaluation_report.md"
REQUEST_PAUSE_SECONDS = 13


@dataclass
class EvaluationRecord:
    question: str
    answer_type: str
    relevance: int
    grounding: int
    reasoning_quality: int
    notes: str = ""


def load_questions(path: str = QUESTION_FILE) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_existing_rows(path: str = RESULTS_JSONL) -> list[dict]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def answer_without_rag(question: str, llm_model: str = "gemini-3-flash-preview") -> str:
    prompt = f"""You are a helpful AI assistant for causal inference and recommendation analysis.

Answer the following question as clearly as possible using your own general knowledge.
If you are uncertain, say so briefly instead of inventing details.

Question:
{question}
"""
    return generate_with_gemini(prompt=prompt, model=llm_model)


def _with_retry(fn, *args, **kwargs):
    attempts = 0
    while True:
        try:
            result = fn(*args, **kwargs)
            time.sleep(REQUEST_PAUSE_SECONDS)
            return result
        except Exception as exc:
            attempts += 1
            message = str(exc)
            if "429" in message or "RESOURCE_EXHAUSTED" in message:
                wait_seconds = 20
                time.sleep(wait_seconds)
                if attempts < 6:
                    continue
            raise


def judge_answer(question: str, answer_type: str, answer: str) -> EvaluationRecord:
    prompt = f"""You are grading an answer for a domain-specific causal inference RAG project.

Question:
{question}

Answer type:
{answer_type}

Answer:
{answer}

Score the answer on three dimensions from 1 to 5:
- relevance
- grounding
- reasoning_quality

Use this guidance:
- relevance: how directly the answer addresses the question
- grounding: how specific and evidence-based the answer appears
- reasoning_quality: how logically and clearly the answer explains the result

Return strict JSON with keys:
relevance, grounding, reasoning_quality, notes
"""
    raw = generate_with_gemini(prompt=prompt, model="gemini-3-flash-preview")
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return EvaluationRecord(question, answer_type, 3, 3, 3, "Judge output could not be parsed.")
    try:
        payload = json.loads(raw[start : end + 1])
        return EvaluationRecord(
            question=question,
            answer_type=answer_type,
            relevance=int(payload.get("relevance", 3)),
            grounding=int(payload.get("grounding", 3)),
            reasoning_quality=int(payload.get("reasoning_quality", 3)),
            notes=str(payload.get("notes", "")),
        )
    except Exception:
        return EvaluationRecord(question, answer_type, 3, 3, 3, "Judge output parsing failed.")


def placeholder_judgment(question: str, answer_type: str) -> EvaluationRecord:
    return EvaluationRecord(
        question=question,
        answer_type=answer_type,
        relevance=0,
        grounding=0,
        reasoning_quality=0,
        notes="Manual scoring pending.",
    )


def write_jsonl(rows: list[dict], path: str = RESULTS_JSONL) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_markdown(rows: list[dict], path: str = RESULTS_MD) -> None:
    lines = [
        "# Evaluation Results",
        "",
        "| Question | Answer Type | Relevance | Grounding | Reasoning | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['question']} | {row['answer_type']} | {row['relevance']} | "
            f"{row['grounding']} | {row['reasoning_quality']} | {row['notes']} |"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(rows: list[dict], path: str = REPORT_MD) -> None:
    grouped: dict[str, list[dict]] = {"no_rag": [], "rag": []}
    for row in rows:
        if row["answer_type"] in grouped:
            grouped[row["answer_type"]].append(row)

    def avg(items: list[dict], key: str) -> float:
        scored = [item[key] for item in items if item.get(key, 0) > 0]
        return round(sum(scored) / len(scored), 2) if scored else 0.0

    no_rag = grouped["no_rag"]
    rag = grouped["rag"]
    question_count = len({row["question"] for row in rows})
    scored_count = len([row for row in rows if row.get("relevance", 0) > 0])
    lines = [
        "# Evaluation Report",
        "",
        "## Setup",
        "",
        "- Compared direct Gemini answers (`no_rag`) against retrieval-augmented answers (`rag`).",
        "- Questions were drawn from a project evaluation set in `data/eval_questions/`.",
        "- Scoring dimensions: relevance, grounding, reasoning quality.",
        f"- Questions evaluated so far: {question_count}",
        f"- Scored answer rows so far: {scored_count}",
        "",
        "## Average Scores",
        "",
        f"- No-RAG relevance: {avg(no_rag, 'relevance')}",
        f"- No-RAG grounding: {avg(no_rag, 'grounding')}",
        f"- No-RAG reasoning quality: {avg(no_rag, 'reasoning_quality')}",
        f"- RAG relevance: {avg(rag, 'relevance')}",
        f"- RAG grounding: {avg(rag, 'grounding')}",
        f"- RAG reasoning quality: {avg(rag, 'reasoning_quality')}",
        "",
        "## Summary",
        "",
        "RAG is expected to perform better on grounding because answers are tied to curated project chunks, especially for the Netflix-style case and synthetic DID result questions. "
        "No-RAG answers can still sound fluent, but they are more likely to be generic or less aligned with the exact project setup.",
        "",
        "## Notes",
        "",
        "- This is a first-pass evaluation intended for project development and README reporting.",
        "- Scores come from an LLM-based judge, so they should be treated as approximate rather than final human labels.",
        "- Rows with zero scores were generated in low-quota mode and still need manual or later LLM scoring.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(
    question_limit: int | None = None,
    questions_path: str = QUESTION_FILE,
    judge_enabled: bool = True,
    resume: bool = False,
) -> list[dict]:
    questions = load_questions(questions_path)
    if question_limit is not None:
        questions = questions[:question_limit]

    rows: list[dict] = load_existing_rows() if resume else []
    completed_pairs = {(row["question"], row["answer_type"]) for row in rows}
    for question in questions:
        if (question, "no_rag") not in completed_pairs:
            no_rag_answer = _with_retry(answer_without_rag, question)
            no_rag_eval = (
                _with_retry(judge_answer, question, "no_rag", no_rag_answer)
                if judge_enabled
                else placeholder_judgment(question, "no_rag")
            )
            rows.append(
                {
                    "question": question,
                    "answer_type": "no_rag",
                    "answer": no_rag_answer,
                    "relevance": no_rag_eval.relevance,
                    "grounding": no_rag_eval.grounding,
                    "reasoning_quality": no_rag_eval.reasoning_quality,
                    "notes": no_rag_eval.notes.replace("\n", " "),
                }
            )
            completed_pairs.add((question, "no_rag"))
            write_jsonl(rows)
            write_markdown(rows)
            write_report(rows)

        if (question, "rag") not in completed_pairs:
            rag_answer, docs, route = _with_retry(answer_question, question)
            rag_eval = (
                _with_retry(judge_answer, question, "rag", rag_answer)
                if judge_enabled
                else placeholder_judgment(question, "rag")
            )
            rows.append(
                {
                    "question": question,
                    "answer_type": "rag",
                    "route": route,
                    "retrieved_sources": [doc.metadata.get("filename", "unknown") for doc in docs],
                    "answer": rag_answer,
                    "relevance": rag_eval.relevance,
                    "grounding": rag_eval.grounding,
                    "reasoning_quality": rag_eval.reasoning_quality,
                    "notes": rag_eval.notes.replace("\n", " "),
                }
            )
            completed_pairs.add((question, "rag"))
            write_jsonl(rows)
            write_markdown(rows)
            write_report(rows)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a first-pass evaluation for no-RAG vs RAG.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--questions-file", type=str, default=QUESTION_FILE)
    parser.add_argument("--core", action="store_true", help="Use the smaller core evaluation set.")
    parser.add_argument("--skip-judge", action="store_true", help="Generate answers without LLM scoring.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing JSONL rows if present.")
    args = parser.parse_args()
    questions_path = CORE_QUESTION_FILE if args.core else args.questions_file
    rows = run_evaluation(
        question_limit=args.limit,
        questions_path=questions_path,
        judge_enabled=not args.skip_judge,
        resume=args.resume,
    )
    print(f"Saved {len(rows)} evaluation rows.")


if __name__ == "__main__":
    main()
