from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from langchain_core.documents import Document


@dataclass
class DatasetBuilder:
    """Builds corpus/query JSONL files and train/test TSV labels, written to GCS.

      - corpus.jsonl: [{"_id": ..., "text": ..., "doc_id": ...}, ...]
      - query.jsonl:  [{"_id": ..., "text": ..., "doc_id": ...}, ...]
      - train.tsv / test.tsv: columns [corpus-id, query-id, score]
        (score=1 for positive pairs)
    """

    processed_tuning_uri: str
    timestamp: str

    def build_frames(
        self,
        chunks: list[Document],
        generated_queries: list[Document],
        train_fraction: float = 0.8,
        seed: int = 7,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create corpus/query dataframes and split label dataframe."""
        corpus_df = pd.DataFrame(
            {
                "_id": [f"text_{idx}" for idx, _ in enumerate(chunks)],
                "text": [chunk.page_content for chunk in chunks],
                # doc_id corresponds to original page number (stored in metadata)
                "doc_id": [chunk.metadata.get("page") for chunk in chunks],
            }
        )

        query_df = pd.DataFrame(
            {
                "_id": [f"query_{idx}" for idx in range(len(generated_queries))],
                "text": [query.page_content for query in generated_queries],
                "doc_id": [query.metadata.get("page") for query in generated_queries],
            }
        )

        # Positive labels: join corpus/query on doc_id (page id)
        score_df = corpus_df.merge(query_df, on="doc_id")
        score_df = score_df.rename(columns={"_id_x": "corpus-id", "_id_y": "query-id"})
        score_df = score_df.drop(columns=["doc_id", "text_x", "text_y"])
        score_df["score"] = 1

        # Random split into train/test
        train_df = score_df.sample(frac=train_fraction, random_state=seed)
        test_df = score_df.drop(train_df.index)

        return corpus_df, query_df, score_df, train_df, test_df

    def write_to_gcs(
        self,
        corpus_df: pd.DataFrame,
        query_df: pd.DataFrame,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict:
        """Write the dataset artifacts to GCS.

        Pandas can write to `gs://...` directly when `gcsfs` is installed.
        """
        base = f"{self.processed_tuning_uri}/{self.timestamp}"

        corpus_path = f"{base}/corpus.jsonl"
        query_path = f"{base}/query.jsonl"
        train_path = f"{base}/train.tsv"
        test_path = f"{base}/test.tsv"

        corpus_df.to_json(corpus_path, orient="records", lines=True)
        query_df.to_json(query_path, orient="records", lines=True)

        train_df.to_csv(train_path, sep="\t", header=True, index=False)
        test_df.to_csv(test_path, sep="\t", header=True, index=False)

        return {
            "corpus_path": corpus_path,
            "query_path": query_path,
            "train_label_path": train_path,
            "test_label_path": test_path,
        }
