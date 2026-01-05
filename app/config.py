from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import os
@dataclass(frozen=True)
class AppConfig:
    """All configuration needed to run the embeddings tuning flow.

    Notes:
      - `bucket_uri` must be a **gs://** URI.
      - `raw_data_uri` is where the PDF lives (also **gs://**).
      - Document AI uses a *location* like "us" or "eu" (not "us-central1").
        We derive it from the Vertex region by taking the first segment.
    """

    project_id: str = os.getenv("PROJECT_ID")
    region: str = os.getenv("REGION", "us-central1")
    bucket_uri: str = os.getenv("BUCKET_URI")

    raw_data_uri: str = "gs://github-repo/embeddings/get_started_with_embedding_tuning"
    pdf_name: str = "goog-10-k-2023.pdf"

    # Dataset creation
    chunk_size: int = 2500
    chunk_overlap: int = 250
    num_questions_per_chunk: int = 3

    # Tuning / training parameters (from the original tutorial)
    batch_size: int = 32
    training_accelerator_type: str = "NVIDIA_TESLA_T4"
    training_machine_type: str = "n1-standard-16"
    base_model_version_id: str = "text-embedding-005"

    # Endpoint deployment parameters
    prediction_accelerator_type: str = "NVIDIA_TESLA_A100"
    prediction_accelerator_count: int = 1
    prediction_machine_type: str = "a2-highgpu-1g"

    # Pipeline template (KFP package hosted by Google)
    template_uri: str = (
        "https://us-kfp.pkg.dev/ml-pipeline/llm-text-embedding/"
        "tune-text-embedding-model/v1.1.1"
    )

    # Timestamp for output folder versioning
    timestamp: str = datetime.now().strftime("%Y%m%d%H%M%S")

    @property
    def docai_location(self) -> str:
        """Document AI API location, e.g. 'us' for 'us-central1'."""
        return self.region.split("-")[0]

    @property
    def processed_data_uri(self) -> str:
        return f"{self.bucket_uri}/data/processed"

    @property
    def prepared_data_uri(self) -> str:
        return f"{self.bucket_uri}/data/prepared"

    @property
    def processed_ocr_uri(self) -> str:
        return f"{self.bucket_uri}/data/processed/ocr"

    @property
    def processed_tuning_uri(self) -> str:
        return f"{self.bucket_uri}/data/processed/tuning"

    @property
    def pipeline_root(self) -> str:
        return f"{self.bucket_uri}/pipelines"

    @property
    def pdf_gcs_path(self) -> str:
        """Full gs:// path to the input PDF."""
        return f"{self.raw_data_uri}/{self.pdf_name}"
    
    def validate(self) -> None:
        """
        Validate that all required configuration values are present.
        Raises a clear error early if something is missing.
        """
        missing = []
        if not self.project_id:
            missing.append("PROJECT_ID")
        if not self.bucket_uri:
            missing.append("BUCKET_URI")

        if missing:
            raise RuntimeError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                f"Did you create a .env file?"
            )
