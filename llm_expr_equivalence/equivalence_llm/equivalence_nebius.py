import json
import time
from datetime import datetime

from openai import OpenAI
from openai.types import Batch, FileObject

from llm_expr_equivalence.config.settings import get_settings, logger
from llm_expr_equivalence.db.data_loader import DataLoader
from llm_expr_equivalence.equivalence_llm.equivalence_base import LLMEquivalenceBase
from llm_expr_equivalence.equivalence_llm.evaluate import LLMEvaluateResultsBase
from llm_expr_equivalence.equivalence_llm.prompts import equivalence_prompt
from llm_expr_equivalence.schemas.common import Trichotomy


class LLMEquivalenceNebius(LLMEquivalenceBase):
    """
    Nebius studio implementation for mathematical expression equivalence checking.
    """

    def __init__(self, model_name: str):
        """
        Initialize the Nebius hosted LLM equivalence checker.
        Args:
            model_name: The name of the model to use
        Sets up both regular and batch clients for OpenAI API calls.
        """

        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(
            base_url="https://api.tokenfactory.nebius.com/v1/",
            api_key=get_settings().nebius_api_key.get_secret_value(),
        )
        self.batch_input_file: FileObject | None = None
        self.batch_process: Batch | None = None
        self.started: str | None = None
        self.in_filename: str | None = None
        self.out_filename: str | None = None

    def __equal__(self, expr1, expr2) -> Trichotomy:
        """
        Compare two expressions for equivalence using OpenAI's chat completion.
        Args:
            expr1: First expression to compare
            expr2: Second expression to compare
        Returns:
            Trichotomy value representing equivalence result
        """

        completion = self.client.chat.completions.create(
            extra_body={},
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": equivalence_prompt(expr1, expr2),
                }
            ]
        )
        logger.info(f"Completion: {completion}")
        print(completion)
        reasoning = getattr(completion.choices[0].message, "reasoning_content", None)
        if reasoning:
            logger.info(f"Reasoning: {reasoning}")

        result = LLMEvaluateResultsBase.extract_result(completion.choices[0].message.content)
        if result is None:
            raise Exception("Failed to extract result from OpenAPI completion")

        return result

    def to_batch_json(self, hash: str, expr1: str, expr2: str) -> dict:
        """
        Convert expressions comparison request to OpenAI batch API format.
        Args:
            hash: Unique identifier for the comparison
            expr1: First expression in latex format
            expr2: Second expression in latex format
        Returns:
            Dictionary formatted for OpenAI's batch processing API
        """

        # Reference: https://platform.openai.com/docs/guides/batch
        return {
            "custom_id": hash,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": equivalence_prompt(expr1, expr2)},
                ],
                "max_completion_tokens": 10000,
                "response_format": {"type": "json_object"},
                "reasoning_effort":"high"
            }
        }

    def submit_batch(self, data: DataLoader):
        """
        Submit a batch of expression comparisons to OpenAI.
        Args:
            data: DataLoader containing expressions to compare
        Creates a JSONL file with requests and submits it for batch processing.
        Stores batch processing details for later retrieval.
        """

        self.started = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.in_filename = f"nebius-batch-{self.started}.jsonl"

        with open(self.in_filename, "w") as f:
            for h, e1, e2, _ in data.data_recs():
                f.write(f"{json.dumps(self.to_batch_json(h, e1["ltx"], e2["ltx"]))}\n")

        # upload batch file
        self.batch_input_file = self.client.files.create(
            file=open(self.in_filename, "rb"),
            purpose="batch"
        )
        logger.info(f"Uploading batch from {self.in_filename}: {self.batch_input_file} ...")

        # create batch processing
        if self.batch_input_file is not None:
            self.batch_process = self.client.batches.create(
                input_file_id=self.batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "test batch llm equivalence"
                }
            )
            logger.info(f"Batch processing created: {self.batch_process}")

    def get_batch_result(self):
        """
        Retrieve and process results from a submitted batch job.
        Returns:
            Trichotomy value representing batch processing results
        Polls the API until batch processing is complete and saves results to file.
        Raises exception if batch processing fails.
        """

        if self.batch_process is None:
            raise Exception("Batch processing not created yet")

        batch = self.client.batches.retrieve(self.batch_process.id)
        started = time.time()
        while batch.status not in ["expired", "completed"]:
            time.sleep(60)
            elapsed = int(time.time() - started)
            logger.info(f"Waiting for batch completion: {elapsed}s elapsed, status: {batch.status}")
            batch = self.client.batches.retrieve(self.batch_process.id)

        logger.info(f"Batch completed: {batch}")

        if batch.output_file_id is None:
            raise Exception("Batch processing failed")

        file_response = self.client.files.content(batch.output_file_id)
        self.out_filename = f"nebius-batch-{self.started}-llm-result.jsonl"
        with open(self.out_filename, "w") as f:
            f.write(file_response.text)
