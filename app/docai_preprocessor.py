from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from langchain_core.documents import Document
from langchain_core.document_loaders import Blob
from langchain_google_community import DocAIParser


@dataclass
class DocAIPreprocessor:
    """Runs Document AI OCR on a PDF in GCS and returns LangChain Documents.

      - Create an OCR Processor
      - Run `DocAIParser.docai_parse()` to process the PDF
      - Poll until it finishes
      - Read results back as LangChain `Document` objects (one per page)

    You can re-use a processor name if desired, but for tutorial simplicity we create a new one.
    """

    project_id: str
    location: str  # Document AI location: "us", "eu", ...
    gcs_output_path: str  # Where DocAI writes OCR output artifacts
    processor_display_name: str

    def create_ocr_processor(self) -> documentai.Processor:
        """Create a Document AI OCR processor."""
        client_options = ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        client = documentai.DocumentProcessorServiceClient(client_options=client_options)

        # Parent is the project+location in which processors live.
        parent = client.common_location_path(self.project_id, self.location)

        return client.create_processor(
            parent=parent,
            processor=documentai.Processor(
                display_name=self.processor_display_name,
                type_="OCR_PROCESSOR",
            ),
        )

    def parse_pdf_from_gcs(self, pdf_gcs_path: str) -> List[Document]:
        """OCR a PDF in GCS and return parsed pages as LangChain Documents."""
        processor = self.create_ocr_processor()

        # Blob is how the LangChain DocAI loader identifies the input file.
        blob = Blob.from_path(path=pdf_gcs_path)

        parser = DocAIParser(
            processor_name=processor.name,
            location=self.location,
            gcs_output_path=self.gcs_output_path,
        )

        # Kick off async processing.
        operations = parser.docai_parse([blob])

        # Poll until processing completes. (The underlying API is async.)
        while True:
            if parser.is_running(operations):
                print("Waiting for Document AI to finish OCR...")
                time.sleep(10)
            else:
                print("Document AI successfully processed the PDF.")
                break

        # Fetch results and convert them to LangChain Document objects.
        results = parser.get_results(operations)
        docs = list(parser.parse_from_results(results))

        # NOTE: Each doc is typically a page; each has `page_content` and metadata.
        return docs
