import json
import time
from datetime import datetime

from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI

from llm_expr_equivalence.config.settings import get_settings, logger
from llm_expr_equivalence.db.data_loader import DataLoader
from llm_expr_equivalence.equivalence_llm.equivalence_base import LLMEquivalenceBase
from llm_expr_equivalence.equivalence_llm.prompts import equivalence_prompt
from llm_expr_equivalence.schemas.common import Trichotomy


class LLMEquivalenceOpenAI(LLMEquivalenceBase):
    """
    OpenAI-based implementation for mathematical expression equivalence checking.

    This class provides functionality to determine if mathematical expressions are equivalent
    using OpenAI's language models. It supports both single comparisons and batch processing
    for efficiency when handling multiple expressions.

    Features:
    - Single expression pair comparison via OpenAI Chat API
    - Batch processing support for multiple comparisons
    - Result retrieval and conversion to pandas DataFrame
    - Automatic handling of OpenAI API rate limits and batch status monitoring

    Inherits from:
        LLMEquivalenceBase: Base class for LLM-based equivalence checking

    Key Components:
    - Regular client for individual queries
    - Batch client for processing multiple comparisons
    - File handling for batch processing input/output
    - Result parsing and formatting utilities
    """

    def __init__(self, model_name: str):
        """
        Initialize the OpenAI LLM equivalence checker.
        Args:
            model_name: The name of the OpenAI model to use
        Sets up both regular and batch clients for OpenAI API calls.
        """

        super().__init__()
        self.model_name = model_name
        self.client = ChatOpenAI(
            api_key=get_settings().openai_api_key,
            model_name=model_name,
            temperature=0,
        )
        self.batch_client = OpenAI(
            api_key=get_settings().openai_api_key,
        )
        self.batch_input_file = None
        self.batch_process = None
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

        messages = [HumanMessage(content=equivalence_prompt(expr1, expr2))]
        response = self.client.invoke(messages)
        return Trichotomy(response.content.strip())

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
                "max_tokens": 1000,
            },
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
        self.in_filename = f"openai-batch-{self.started}.jsonl"

        with open(self.in_filename, "w") as f:
            for h, e1, e2, _ in data.data_recs():
                f.write(f"{json.dumps(self.to_batch_json(h, e1["ltx"], e2["ltx"]))}\n")

        # upload batch file
        self.batch_input_file = self.batch_client.files.create(
            file=open(self.in_filename, "rb"),
            purpose="batch"
        )
        logger.info(f"Uploading batch from {self.in_filename}: {self.batch_input_file} ...")

        # create batch processing
        if self.batch_input_file is not None:
            self.batch_process = self.batch_client.batches.create(
                input_file_id=self.batch_input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "test batch llm equivalence"
                }
            )
            logger.info(f"Batch processing created: {self.batch_process}")

    def get_batch_result(self) -> Trichotomy:
        """
        Retrieve and process results from a submitted batch job.
        Returns:
            Trichotomy value representing batch processing results
        Polls the API until batch processing is complete and saves results to file.
        Raises exception if batch processing fails.
        """

        if self.batch_process is None:
            raise Exception("Batch processing not created yet")

        batch = self.batch_client.batches.retrieve(self.batch_process.id)
        started = time.time()
        while batch.status not in ["expired", "completed"]:
            time.sleep(60)
            print(f"\rWaiting: {int(time.time() - started)}", end="", flush=True)
            batch = self.batch_client.batches.retrieve(self.batch_process.id)

        logger.info(f"Batch completed: {batch}")

        if batch.output_file_id is None:
            raise Exception("Batch processing failed")

        file_response = self.batch_client.files.content(batch.output_file_id)
        self.out_filename = f"openai-batch-{self.started}-llm-result.jsonl"
        with open(self.out_filename, "w") as f:
            f.write(file_response.text)
