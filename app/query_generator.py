from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from vertexai.generative_models import GenerativeModel, GenerationConfig
from vertexai.preview import generative_models


@dataclass
class GeminiQueryGenerator:
    """Generates synthetic retrieval queries for each chunk using Gemini.

    Generates *one* question per chunk (default 3 per chunk),
    by prompting Gemini to act as an exam writer and craft questions about the text.

    Return the result as a LangChain `Document`:
      - `page_content` holds the generated query text
      - `metadata` stores the original chunk/page number so we can build labels
    """

    model_name: str = "gemini-2.0-flash"
    max_output_tokens: int = 2048
    temperature: float = 0.9
    top_p: float = 1.0

    def generate(self, chunk: Document, num_questions: int = 3) -> Document:
        """Generate questions (queries) for a given chunk.

        Returns:
          A Document where `page_content` is a single question string.
          If multiple questions are requested, we join them with newlines.
        """
        model = GenerativeModel(self.model_name)

        generation_config = GenerationConfig(
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Tutorial disables safety blocking to avoid unexpected failures.
        # You can tighten this in production.
        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

        prompt = f"""You are an exam writer.
Your task is to create {num_questions} QUESTION(s) that can be answered from the TEXT below.
- Questions should be concise.
- Do not include the answer.
- Do not mention that the question came from a document.
- Return each question on its own line.

TEXT:
{chunk.page_content}
"""

        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

        # Gemini may return numbering/bullets; we normalize to plain lines.
        text = response.text or ""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        lines = [re.sub(r"^[-*\d.\)\s]+", "", ln).strip() for ln in lines]
        lines = [ln for ln in lines if ln]

        # Ensure we return exactly `num_questions` lines if possible.
        if len(lines) > num_questions:
            lines = lines[:num_questions]

        joined = "\n".join(lines) if lines else ""

        return Document(page_content=joined, metadata=dict(chunk.metadata))
