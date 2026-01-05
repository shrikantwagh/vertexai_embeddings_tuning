from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from google.protobuf.json_format import MessageToDict
from google.cloud import aiplatform


@dataclass
class EmbeddingTuningJob:
    """Wraps the Vertex AI KFP pipeline that tunes a text embedding model."""

    project_id: str
    region: str
    bucket_uri: str
    pipeline_root: str
    template_uri: str

    def submit(
        self,
        *,
        corpus_path: str,
        query_path: str,
        train_label_path: str,
        test_label_path: str,
        batch_size: int,
        iterations: int,
        accelerator_type: str,
        machine_type: str,
        base_model_version_id: str,
        display_name: str = "tune-text-embedding",
    ) -> aiplatform.PipelineJob:
        """Submit the pipeline job and block until completion."""
        aiplatform.init(project=self.project_id, location=self.region, staging_bucket=self.bucket_uri)

        params = {
            "batch_size": batch_size,
            "iterations": iterations,
            "accelerator_type": accelerator_type,
            "machine_type": machine_type,
            "base_model_version_id": base_model_version_id,
            "queries_path": query_path,
            "corpus_path": corpus_path,
            "train_label_path": train_label_path,
            "test_label_path": test_label_path,
            "project": self.project_id,
            "location": self.region,
        }

        job = aiplatform.PipelineJob(
            display_name=display_name,
            parameter_values=params,
            template_path=self.template_uri,
            pipeline_root=self.pipeline_root,
            project=self.project_id,
            location=self.region,
        )

        # This call blocks by default until the pipeline finishes.
        #job.run()

        job.run(sync=False)
        print(f"Pipeline submitted: {job.resource_name}")
        print(f"Console: {job._dashboard_uri()}")

        return job

    # --- Helper methods ---

    @staticmethod
    def get_task_by_name(job: aiplatform.PipelineJob, task_name: str):
        """Return a specific pipeline task detail by `task_name`."""
        for task in job.task_details:
            if task.task_name == task_name:
                return task
        raise ValueError(f"Task {task_name} not found in pipeline job.")

    @classmethod
    def get_metrics(cls, job: aiplatform.PipelineJob, task_name: str = "text-embedding-evaluator") -> pd.DataFrame:
        """Fetch evaluation metrics produced by the pipeline."""
        evaluation_task = cls.get_task_by_name(job, task_name)
        metrics = MessageToDict(evaluation_task.outputs["metrics"]._pb)["artifacts"][0]["metadata"]
        return pd.DataFrame([metrics])

    @classmethod
    def get_uploaded_model(cls, job: aiplatform.PipelineJob, task_name: str = "text-embedding-model-uploader") -> aiplatform.Model:
        """Return the uploaded tuned model as an `aiplatform.Model` resource."""
        uploader_task = cls.get_task_by_name(job, task_name)
        upload_metadata = MessageToDict(uploader_task.execution._pb)["metadata"]
        return aiplatform.Model(upload_metadata["output:model_resource_name"])

    @classmethod
    def get_training_output_dir(cls, job: aiplatform.PipelineJob, task_name: str = "text-embedding-trainer") -> str:
        """Return the training output GCS directory produced by the trainer task."""
        trainer_task = cls.get_task_by_name(job, task_name)
        output_artifacts = MessageToDict(trainer_task.outputs["training_output"]._pb)["artifacts"][0]
        return output_artifacts["uri"]
