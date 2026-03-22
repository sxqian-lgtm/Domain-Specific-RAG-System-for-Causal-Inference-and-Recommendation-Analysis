"""Route user questions to the most relevant knowledge source."""

from __future__ import annotations


METHOD_KEYWORDS = {
    "did",
    "difference in differences",
    "ipw",
    "att",
    "ate",
    "confounder",
    "confounding",
    "parallel trends",
    "backdoor",
    "do operator",
    "counterfactual",
    "d separation",
    "collider",
    "regression adjustment",
    "causal effect",
}

PROJECT_KEYWORDS = {
    "netflix",
    "recommendation",
    "recommender",
    "recsys",
    "viewing time",
    "minutes viewed",
    "project",
    "synthetic data",
    "major release",
    "business",
    "experiment",
    "product",
    "analysis",
}


def route_query(question: str) -> str:
    text = question.lower()
    method_hit = any(keyword in text for keyword in METHOD_KEYWORDS)
    project_hit = any(keyword in text for keyword in PROJECT_KEYWORDS)

    if method_hit and project_hit:
        return "mixed"
    if method_hit:
        return "method"
    if project_hit:
        return "project"
    return "mixed"
