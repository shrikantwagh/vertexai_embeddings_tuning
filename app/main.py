from __future__ import annotations

import argparse
import math
import random
import string
from dataclasses import asdict

from google.cloud import aiplatform
from google.cloud import storage
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import vertexai  # Vertex AI SDK (for Gemini + general initialization)

from .config import AppConfig
from .docai_preprocessor import DocAIPreprocessor
from .query_generator import GeminiQueryGenerator
from .dataset_builder import DatasetBuilder
from .tuner import EmbeddingTuningJob
from .retrieval import Retriever


# -----------------------------
# Small GCS helpers
# -----------------------------

def maybe_create_bucket(project_id: str, bucket_uri: str, region: str) -> None:
    """Create a GCS bucket if it doesn't exist.

    gcp shell: `gsutil mb -l {REGION} -p {PROJECT_ID} {BUCKET_URI}`.

    Here we use the Storage client instead, so the whole flow is Python.
    """
    if not bucket_uri.startswith("gs://"):
        raise ValueError("bucket_uri must start with gs://")

    bucket_name = bucket_uri[len("gs://") :].rstrip("/")

    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)

    if bucket.exists(client):
        print(f"GCS bucket already exists: {bucket_uri}")
        return

    # NOTE: region must be a valid GCS location (often same as Vertex region).
    bucket.location = region
    client.create_bucket(bucket)
    print(f"Created GCS bucket: {bucket_uri}")


# -----------------------------
# End-to-end flow
# -----------------------------

def run_end_to_end(cfg: AppConfig, *, create_bucket: bool, force_ocr: bool) -> dict:
    """Run the full embeddings tuning pipeline, then deploy and demo retrieval.

    Returns a dict of resource IDs and output locations you can reuse for retrieval / cleanup.
    """
    if create_bucket:
        maybe_create_bucket(cfg.project_id, cfg.bucket_uri, cfg.region)

    # Initialize Vertex AI libraries (Gemini + Pipelines/Endpoints).
    vertexai.init(project=cfg.project_id, location=cfg.region, staging_bucket=cfg.bucket_uri)
    aiplatform.init(project=cfg.project_id, location=cfg.region, staging_bucket=cfg.bucket_uri)

    # ---- 1) Document AI OCR → LangChain Documents ----
    # Use a deterministic name for the processor.
    processor_display_name = "preprocess-docs-llm-tutorial"

    pre = DocAIPreprocessor(
        project_id=cfg.project_id,
        location=cfg.docai_location,
        gcs_output_path=cfg.processed_ocr_uri,
        processor_display_name=processor_display_name,
        force_ocr=force_ocr,
    )

    docs = pre.parse_pdf_from_gcs(cfg.pdf_gcs_path)
    if not docs:
        raise RuntimeError("Document AI returned zero documents/pages.")

    # ---- 2) Chunking ----
    # We keep the same defaults as the tutorial (chunk_size=2500, overlap=250).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )

    # Each LangChain doc is treated like a page.
    # We attach page number metadata starting at 1.
    document_content = [doc.page_content for doc in docs]
    document_metadata = [{"page": idx} for idx, _ in enumerate(docs, start=1)]
    chunks = splitter.create_documents(document_content, metadatas=document_metadata)

    # ---- 3) Synthetic queries using Gemini ----
    generator = GeminiQueryGenerator()
    generated_queries = []
    for chunk in chunks:
        qdoc = generator.generate(chunk, num_questions=cfg.num_questions_per_chunk)

        # If we requested multiple questions, we split them into separate query docs.
        # This keeps label building straightforward.
        for line in [ln.strip() for ln in (qdoc.page_content or "").splitlines() if ln.strip()]:
            generated_queries.append(type(qdoc)(page_content=line, metadata=dict(qdoc.metadata)))

    if not generated_queries:
        raise RuntimeError("Gemini produced zero queries. Check auth / model access / prompt.")

    # ---- 4) Build corpus/query + labels and write to GCS ----
    builder = DatasetBuilder(processed_tuning_uri=cfg.processed_tuning_uri, timestamp=cfg.timestamp)

    corpus_df, query_df, _, train_df, test_df = builder.build_frames(
        chunks=chunks,
        generated_queries=generated_queries,
        train_fraction=0.8,
        seed=7,
    )

    paths = builder.write_to_gcs(corpus_df, query_df, train_df, test_df)

    # ---- 5) Launch tuning pipeline ----

    iterations = max(1, len(train_df) // max(1, cfg.batch_size))

    tuning = EmbeddingTuningJob(
        project_id=cfg.project_id,
        region=cfg.region,
        bucket_uri=cfg.bucket_uri,
        pipeline_root=cfg.pipeline_root,
        template_uri=cfg.template_uri,
    )

    job = tuning.submit(
        corpus_path=paths["corpus_path"],
        query_path=paths["query_path"],
        train_label_path=paths["train_label_path"],
        test_label_path=paths["test_label_path"],
        batch_size=cfg.batch_size,
        iterations=iterations,
        accelerator_type=cfg.training_accelerator_type,
        machine_type=cfg.training_machine_type,
        base_model_version_id=cfg.base_model_version_id,
    )

    print("Waiting for pipeline job to complete...")
    job.wait()

    metrics_df = tuning.get_metrics(job)
    print("Pipeline metrics:")
    print(metrics_df.to_string(index=False))

    model = tuning.get_uploaded_model(job)
    training_output_dir = tuning.get_training_output_dir(job)

    # ---- 6) Deploy tuned model to endpoint ----
    endpoint = aiplatform.Endpoint.create(
        display_name="tuned_custom_embedding_endpoint",
        description="Endpoint for tuned model embeddings.",
        project=cfg.project_id,
        location=cfg.region,
    )

    endpoint.deploy(
        model,
        accelerator_type=cfg.prediction_accelerator_type,
        accelerator_count=cfg.prediction_accelerator_count,
        machine_type=cfg.prediction_machine_type,
    )

    # ---- 7) Retrieval demo ----
    corpus_text, corpus_embeddings = Retriever.load_corpus_from_training_output(training_output_dir)

    queries = [
        "What about the revenues?",
        "Who is Alphabet?",
        "What about the costs?",
    ]

    retriever = Retriever(project_id=cfg.project_id, region=cfg.region, endpoint_id=endpoint.resource_name)
    output = retriever.top_k_documents(
        queries,
        corpus_text=corpus_text,
        corpus_embeddings=corpus_embeddings,
        k=10,
    )

    print("\nTop‑k retrieval demo (first 3 rows):")
    with pd.option_context("display.max_colwidth", 120):
        print(output.head(3).to_string(index=False))

    return {
        "pipeline_job_resource_name": job.resource_name,
        "model_resource_name": model.resource_name,
        "endpoint_resource_name": endpoint.resource_name,
        "training_output_dir": training_output_dir,
        "dataset_paths": paths,
        "metrics": metrics_df.to_dict(orient="records")[0] if not metrics_df.empty else {},
        "config": asdict(cfg),
    }


def run_retrieve(project_id: str, region: str, endpoint_id: str, training_output_dir: str, k: int) -> None:
    """Load corpus+embeddings from a previous run and query the endpoint."""
    corpus_text, corpus_embeddings = Retriever.load_corpus_from_training_output(training_output_dir)

    retriever = Retriever(project_id=project_id, region=region, endpoint_id=endpoint_id)

    queries = [
        "What about the revenues?", "Who is Alphabet?", "What about the costs?"
        "Who is Alphabet?",
        "What about the costs?",
    ]

    output = retriever.top_k_documents(
        queries,
        corpus_text=corpus_text,
        corpus_embeddings=corpus_embeddings,
        k=k,
    )

    with pd.option_context("display.max_colwidth", 120):
        print(output.head(min(10, k)).to_string(index=False))


def run_cleanup(
    project_id: str,
    region: str,
    endpoint_id: str | None,
    model_id: str | None,
    pipeline_job_id: str | None,
    bucket_uri: str | None,
    delete_bucket_objects: bool,
) -> None:
    """Best-effort cleanup of resources created by the program."""
    aiplatform.init(project=project_id, location=region)

    # Endpoint
    if endpoint_id:
        try:
            ep = aiplatform.Endpoint(endpoint_id)
            ep.delete(force=True)
            print(f"Deleted endpoint: {endpoint_id}")
        except Exception as e:
            print(f"Failed to delete endpoint {endpoint_id}: {e}")

    # Model
    if model_id:
        try:
            model = aiplatform.Model(model_id)
            model.delete()
            print(f"Deleted model: {model_id}")
        except Exception as e:
            print(f"Failed to delete model {model_id}: {e}")

    # Pipeline job
    if pipeline_job_id:
        try:
            job = aiplatform.PipelineJob.get(pipeline_job_id)
            job.delete()
            print(f"Deleted pipeline job: {pipeline_job_id}")
        except Exception as e:
            print(f"Failed to delete pipeline job {pipeline_job_id}: {e}")

    # Bucket objects (optional)
    if bucket_uri and delete_bucket_objects:
        if not bucket_uri.startswith("gs://"):
            raise ValueError("bucket_uri must start with gs://")

        bucket_name = bucket_uri[len("gs://") :].rstrip("/")
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)

        try:
            blobs = list(bucket.list_blobs())
            for b in blobs:
                b.delete()
            print(f"Deleted {len(blobs)} objects in bucket: {bucket_uri}")
        except Exception as e:
            print(f"Failed to delete bucket objects in {bucket_uri}: {e}")


# -----------------------------
# CLI
# -----------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vertex AI embeddings tuning refactor (program).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # run
    p_run = sub.add_parser("run", help="Run OCR → dataset → tuning → deploy → retrieval demo.")
    p_run.add_argument("--project-id", required=True)
    p_run.add_argument("--region", default="us-us-east1")
    p_run.add_argument("--bucket-uri", required=True)
    p_run.add_argument("--raw-data-uri", default="gs://github-repo/embeddings/get_started_with_embedding_tuning")
    p_run.add_argument("--pdf-name", default="goog-10-k-2023.pdf")
    p_run.add_argument("--chunk-size", type=int, default=2500)
    p_run.add_argument("--chunk-overlap", type=int, default=250)
    p_run.add_argument("--num-questions-per-chunk", type=int, default=3)
    p_run.add_argument("--batch-size", type=int, default=32)
    p_run.add_argument("--create-bucket", type=str, default="false")
    p_run.add_argument("--force-ocr", type=str, default="false", help="Force re-running OCR even if output exists.")

    # retrieve
    p_ret = sub.add_parser("retrieve", help="Run retrieval demo given endpoint and training output dir.")
    p_ret.add_argument("--project-id", required=True)
    p_ret.add_argument("--region", default="us-east1")
    p_ret.add_argument("--endpoint-id", required=True, help="Full endpoint resource name or ID.")
    p_ret.add_argument("--training-output-dir", required=True, help="GCS directory from pipeline training output.")
    p_ret.add_argument("--k", type=int, default=10)

    # cleanup
    p_cl = sub.add_parser("cleanup", help="Delete endpoint/model/pipeline job (best-effort).")
    p_cl.add_argument("--project-id", required=True)
    p_cl.add_argument("--region", default="us-east1")
    p_cl.add_argument("--endpoint-id", default=None)
    p_cl.add_argument("--model-id", default=None)
    p_cl.add_argument("--pipeline-job-id", default=None)
    p_cl.add_argument("--bucket-uri", default=None)
    p_cl.add_argument("--delete-bucket-objects", type=str, default="false")

    return parser


def str_to_bool(v: str) -> bool:
    return v.strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        cfg = AppConfig(
            project_id=args.project_id,
            region=args.region,
            bucket_uri=args.bucket_uri,
            raw_data_uri=args.raw_data_uri,
            pdf_name=args.pdf_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            num_questions_per_chunk=args.num_questions_per_chunk,
            batch_size=args.batch_size,
        )

        cfg.validate()

        result = run_end_to_end(cfg, create_bucket=str_to_bool(args.create_bucket), force_ocr=str_to_bool(args.force_ocr))
        print("\nRun complete. Save these for reuse / cleanup:")
        for k, v in result.items():
            if k == "metrics":
                continue
            print(f"- {k}: {v}")

    elif args.cmd == "retrieve":
        run_retrieve(
            project_id=args.project_id,
            region=args.region,
            endpoint_id=args.endpoint_id,
            training_output_dir=args.training_output_dir,
            k=args.k,
        )

    elif args.cmd == "cleanup":
        run_cleanup(
            project_id=args.project_id,
            region=args.region,
            endpoint_id=args.endpoint_id,
            model_id=args.model_id,
            pipeline_job_id=args.pipeline_job_id,
            bucket_uri=args.bucket_uri,
            delete_bucket_objects=str_to_bool(args.delete_bucket_objects),
        )


if __name__ == "__main__":
    main()
