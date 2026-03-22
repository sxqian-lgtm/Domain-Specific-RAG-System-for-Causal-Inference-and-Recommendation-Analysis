"""Gemini client setup for answer generation."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def generate_with_gemini(
    prompt: str,
    model: str = "gemini-3-flash-preview",
    thinking_level: str = "low",
) -> str:
    load_dotenv()

    gemini_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or os.getenv("API")
    )
    if not gemini_key:
        raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY in your .env file.")

    client = genai.Client(api_key=gemini_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level)
        ),
    )
    return response.text or ""
