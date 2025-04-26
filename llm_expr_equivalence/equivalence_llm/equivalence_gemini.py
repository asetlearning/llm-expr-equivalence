import json
import os
import time
from datetime import datetime

import fsspec
from google import genai
from google.auth.transport.requests import Request
from google.cloud import storage
from google.genai import types
from google.genai.types import BatchJob, CreateBatchJobConfig
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

from llm_expr_equivalence.config.settings import get_settings, logger
from llm_expr_equivalence.db.data_loader import DataLoader
from llm_expr_equivalence.equivalence_llm.equivalence_base import (
    LLMEquivalenceBase,
    LLMEquivalenceBatchBase,
)
from llm_expr_equivalence.equivalence_llm.evaluate import LLMEvaluateResultsBase
from llm_expr_equivalence.equivalence_llm.prompts import equivalence_prompt
from llm_expr_equivalence.schemas.common import Trichotomy


class LLMEquivalenceGemini(LLMEquivalenceBase):

    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        """
        Initialize the OpenAI LLM equivalence checker.
        Args:
            model_name: The name of the OpenAI model to use
        Sets up both regular and batch clients for OpenAI API calls.
        """

        super().__init__()
        self.model_name = model_name
        self.client = genai.Client(
            api_key=get_settings().gemini_api_key.get_secret_value()
        )

        # Optional config
        self.config = types.GenerateContentConfig(
            system_instruction="You are a mathematician and your goal to decide if two expressions are equivalent.",
            max_output_tokens=400,
            top_k=2,
            top_p=0.5,
            temperature=0,
            response_mime_type="application/json",
            #      stop_sequences= ['\n'],
            seed=42,
        )

    def __equal__(self, expr1, expr2) -> Trichotomy | None:
        response = self.client.models.generate_content(
            model=self.model_name,
            config=self.config,
            contents=equivalence_prompt(expr1, expr2),
        )

        return LLMEvaluateResultsBase.extract_result(response.text)


class LLMEquivalenceBatchGemini(LLMEquivalenceBatchBase):
    """
    Facilitates batch processing of equivalence checks for LLMs using Google
    Cloud Vertex AI and Google Cloud Storage buckets.

    Source example: https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction.ipynb
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
        self.storage_client = storage.Client.from_service_account_json(
            get_settings().google_cloud_service_storage_creds_json
        )
        self.creds = self.get_creds()
        self.client = genai.Client(
            vertexai=True,
            project=get_settings().google_cloud_project,
            location=get_settings().google_cloud_location,
            credentials=self.creds,
        )
        self.gcs_batch_job: BatchJob | None = None

    def get_creds(self):
        """Converts clients secret json to a credential object.

        Saves generated token file to minimize the use of the consent screen.
        """

        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        scopes = ["https://www.googleapis.com/auth/cloud-platform"]
        if os.path.exists(get_settings().google_vertex_access_token_json):
            creds = Credentials.from_authorized_user_file(
                get_settings().google_vertex_access_token_json, scopes=scopes
            )

        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    get_settings().google_vertex_client_secret_json, scopes=scopes
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(get_settings().google_vertex_access_token_json, "w") as token:
                token.write(creds.to_json())
        return creds

    def to_batch_json(self, hash: str, expr1: str, expr2: str) -> dict:
        """
        Converts the input parameters into format acceptable by Vertex AI batch processing

        Description: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
        """
        return {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": equivalence_prompt(expr1, expr2),
                            }
                        ],
                    }
                ],
                # TODO: see if it useful
                # "systemInstruction": {
                #   "role": "system", # Ignored so can be anythibg
                #   "parts": [
                #     {
                #       "text": "You are a mathematician and your goal to decide if two expressions are equivalent."
                #     }
                #   ]
                # },
                "generationConfig": {
                    "temperature": 0,
                    "topP": 0.5,
                    "topK": 2,
                    #    "candidateCount": integer,
                    "maxOutputTokens": 400,
                    #    "presencePenalty": float,
                    #    "frequencyPenalty": float,
                    #    "stopSequences": [
                    #      string
                    #    ],
                    "responseMimeType": "application/json",
                    #    "responseSchema": schema,
                    #    "seed": integer,
                    #    "responseLogprobs": boolean,
                    #    "logprobs": integer,
                    #    "audioTimestamp": boolean
                },
                "labels": {"llm-equivalence-hash": hash},
            }
        }

    def submit_batch(
        self,
        data: DataLoader,
        bucket_name: str = get_settings().google_cloud_input_bucket,
    ):

        self.started = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.in_filename = f"google-batch-{self.started}.jsonl"
        self.in_blob = f"equivalent_exprs_batch-{self.started}.jsonl"

        with open(self.in_filename, "w") as f:
            for h, e1, e2, _ in data.data_recs():
                f.write(f"{json.dumps(self.to_batch_json(h, e1['ltx'], e2['ltx']))}\n")

        # Upload data to a bucket
        bucket = self.storage_client.get_bucket(bucket_name)
        blob = bucket.blob(self.in_blob)
        blob.upload_from_filename(self.in_filename)
        logger.info(f"Uploaded batch from {self.in_filename} to {blob.path} ...")

        # create batch processing
        logger.info(f"Using model: {self.model_name}")
        self.input_data = (
            f"gs://{get_settings().google_cloud_input_bucket}/{self.in_blob}"
        )
        self.ouput_bucket = f"gs://{get_settings().google_cloud_output_bucket}"  # @param {type:"string"} output bucket
        self.gcs_batch_job = self.client.batches.create(
            model=self.model_name,
            src=self.input_data,
            config=CreateBatchJobConfig(
                dest=self.ouput_bucket,
                display_name=f"llm-equivalence-batch-{self.started}",
            ),
        )
        logger.info(f"Batch processing created: {self.gcs_batch_job.name}")

    # Status can be viewed in VertexAI console:
    # https://console.cloud.google.com/vertex-ai/batch-predictions?project=cs-poc-tpmqhq7sbs86nncrcjhlnsh
    def get_batch_result(self):
        if self.gcs_batch_job is None:
            raise Exception("Batch processing not created yet")

        # Refresh the job until complete
        started = time.time()
        IN_PROGRESS_STATES = [
            "JOB_STATE_DRAINING",
            "JOB_STATE_RESOURCE_CLEANING_UP",
            "JOB_STATE_QUEUED",
            "JOB_STATE_SCHEDULED",
            "JOB_STATE_PENDING",
            "JOB_STATE_RUNNING",
        ]

        while self.gcs_batch_job.state in IN_PROGRESS_STATES:
            print(f"\rWaiting: {int(time.time() - started)}", end="", flush=True)
            time.sleep(10)
            self.gcs_batch_job = self.client.batches.get(name=self.gcs_batch_job.name)

        # Check if the job succeeds
        if self.gcs_batch_job.state == "JOB_STATE_SUCCEEDED":
            logger.info(f"Batch completed: {self.gcs_batch_job.name}")

            # Download results files from cloud
            self.out_filename = f"gemini-batch-{self.started}-llm-result.jsonl"

            # Get the path to results blob
            fs = fsspec.filesystem("gcs")
            file_paths = fs.glob(f"gs://{get_settings().google_cloud_output_bucket}/*/predictions.jsonl")

            if file_paths:
                # TODO: fix multiple file results
                # if more then one than take the last in the list. It should be latest?
                file_name = file_paths[-1]
            else:
                raise Exception(f"No results found in {self.gcs_batch_job.dest.gcs_uri}")

            logger.info(
                f"Downloading results from {file_name} to {self.out_filename} ..."
            )
            blob = storage.Blob.from_string(file_name, client=self.storage_client)
            blob.download_to_filename(self.out_filename)

            # Load the JSONL file into a DataFrame
            # df = pd.read_json(f"gs://{file_paths[0]}", lines=True)
            #
            # df = df.join(pd.json_normalize(df["response"], "candidates"))

            # file_response = self.batch_client.files.content(batch.output_file_id)
            # self.out_filename = f"google-batch-{self.started}-llm-result.jsonl"
            # with open(self.out_filename, "w") as f:
            #     f.write(file_response.text)

        else:
            raise Exception(f"Batch processing failed: {self.gcs_batch_job.error}")
