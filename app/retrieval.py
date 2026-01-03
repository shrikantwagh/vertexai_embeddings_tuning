from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from google.cloud import aiplatform


@dataclass
class Retriever:
    """Runs a simple top‑k retrieval demo using a tuned embedding endpoint.

    The pipeline writes:
      - corpus_text.jsonl     (the corpus texts)
      - corpus_custom.jsonl   (the tuned embeddings for the corpus)

    We embed the query texts using the deployed endpoint and return top‑k matches
    by dot-product similarity (the tutorial assumes normalized vectors).
    """

    project_id: str
    region: str
    endpoint_id: str

    def _endpoint(self) -> aiplatform.Endpoint:
        aiplatform.init(project=self.project_id, location=self.region)
        return aiplatform.Endpoint(self.endpoint_id)

    @staticmethod
    def get_top_k_scores(query_embedding: np.ndarray, corpus_embeddings: pd.DataFrame, k: int = 10) -> pd.DataFrame:
        """Return the top-k indices per query by dot product similarity."""
        # corpus_embeddings is a dataframe where each row is an embedding vector.
        similarity = corpus_embeddings.dot(query_embedding.T)
        topk_index = pd.DataFrame({c: v.nlargest(n=k).index for c, v in similarity.items()})
        return topk_index

    def embed_queries(
        self,
        query_texts: List[str],
        *,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: str = "",
    ) -> np.ndarray:
        """Call the endpoint to generate embeddings for the given queries."""
        endpoint = self._endpoint()

        # Each instance follows the Vertex embedding prediction schema used in the notebook.
        instances = [{"content": t, "task_type": task_type, "title": title} for t in query_texts]

        response = endpoint.predict(instances=instances)

        # Endpoint returns a list-of-lists. Convert to ndarray shaped [num_queries, dim].
        return np.asarray(response.predictions)

    def top_k_documents(
        self,
        query_texts: List[str],
        *,
        corpus_text: pd.DataFrame,
        corpus_embeddings: pd.DataFrame,
        k: int = 10,
        task_type: str = "RETRIEVAL_DOCUMENT",
        title: str = "",
    ) -> pd.DataFrame:
        """Return a dataframe with each column = query and rows = top-k retrieved corpus texts."""
        query_embedding = self.embed_queries(query_texts, task_type=task_type, title=title)
        topk = self.get_top_k_scores(query_embedding, corpus_embeddings, k=k)

        # Build a results dataframe: each column is the query; cells are retrieved texts.
        return pd.DataFrame.from_dict(
            {query_texts[c]: corpus_text.loc[v.values].values.ravel() for c, v in topk.items()},
            orient="columns",
        )

    @staticmethod
    def load_corpus_from_training_output(training_output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load corpus_text and corpus_custom embeddings from a pipeline output directory."""
        corpus_text = pd.read_json(f"{training_output_dir}/corpus_text.jsonl", lines=True)
        corpus_embeddings = pd.read_json(f"{training_output_dir}/corpus_custom.jsonl", lines=True)
        return corpus_text, corpus_embeddings
