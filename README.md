# Vertex AI — Embeddings Tuning (Python Program)


It demonstrates an end-to-end flow:

1. **OCR & preprocessing** a PDF with **Document AI** (OCR Processor)
2. **Chunking** the extracted text
3. Creating a **synthetic query ↔ chunk** dataset using **Gemini**
4. Launching the **Vertex AI embeddings tuning pipeline**
5. Deploying the tuned embedding model to a **Vertex AI Endpoint**
6. Running a tiny **retrieval demo** (top‑k chunks per query)
7. Optional **cleanup** of cloud resources

> ⚠️ Costs: This can incur charges (Document AI, Vertex AI pipelines, endpoint deployment + GPUs, storage).

---

## Prerequisites

### 1) Google Cloud project + auth

- A Google Cloud project with billing enabled.
- You have permission to use:
  - Vertex AI (Pipelines, Model Registry, Endpoints)
  - Document AI (create processor + run OCR)
  - Cloud Storage (read/write)

Authenticate locally (one of the following):

```bash
gcloud auth application-default login
# or (when running on GCE/Cloud Shell/Colab) ensure ADC is already available
```

### 2) Enable APIs (once per project)

```bash
gcloud services enable aiplatform.googleapis.com documentai.googleapis.com storage.googleapis.com
```

### 3) Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start

### 1) Set environment variables

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
export REGION="us-central1"
export BUCKET_URI="gs://YOUR_BUCKET_NAME"   # must already exist OR set --create-bucket
```

### 2) Run the pipeline end-to-end

This uses the sample PDF referenced by the notebook:
`gs://github-repo/embeddings/get_started_with_embedding_tuning/goog-10-k-2023.pdf`

```bash
python -m app.main run \
  --project-id "$PROJECT_ID" \
  --region "$REGION" \
  --bucket-uri "$BUCKET_URI" \
  --raw-data-uri "gs://github-repo/embeddings/get_started_with_embedding_tuning" \
  --pdf-name "goog-10-k-2023.pdf" \
  --create-bucket false
```

### 3) Retrieval demo only (after a successful run)

If you already have a previous pipeline output directory and an endpoint deployed, you can run:

```bash
python -m app.main retrieve \
  --project-id "$PROJECT_ID" \
  --region "$REGION" \
  --bucket-uri "$BUCKET_URI" \
  --endpoint-id "YOUR_ENDPOINT_ID" \
  --training-output-dir "gs://.../pipelines/.../..." 
```

---

## Cleanup

To delete cloud resources created by this program (endpoint, model, pipeline job, optionally bucket objects):

```bash
python -m app.main cleanup \
  --project-id "$PROJECT_ID" \
  --region "$REGION" \
  --endpoint-id "YOUR_ENDPOINT_ID" \
  --model-id "YOUR_MODEL_ID" \
  --pipeline-job-id "YOUR_PIPELINE_JOB_ID" \
  --bucket-uri "$BUCKET_URI" \
  --delete-bucket-objects false
```

---

## Notes / common troubleshooting

- **Document AI location**: Document AI uses `LOCATION` like `us` (derived from `REGION` by taking the first segment).
  Example: `us-central1` → `us`.
- If the pipeline fails due to quota, try smaller settings:
  - reduce `--batch-size`
  - reduce `--chunk-size`
  - reduce `--num-questions-per-chunk`
- If endpoint deployment is expensive, use CPU machine types (but embeddings prediction may require accelerators depending on model + region).

---

## Project layout

- `app/config.py` — configuration dataclass
- `app/docai_preprocessor.py` — Document AI OCR + parsing to LangChain `Document`
- `app/query_generator.py` — Gemini-based synthetic query generation
- `app/dataset_builder.py` — corpus/query JSONL + train/test TSV writer to GCS
- `app/tuner.py` — Vertex AI pipeline submission + metrics/model/output helpers
- `app/retrieval.py` — top‑k similarity search using tuned embeddings
- `app/main.py` — CLI entrypoint

---

## Disclaimer

This is a  tutorial and not a production ready code. 
You still need appropriate Google Cloud permissions and you are responsible for costs.
