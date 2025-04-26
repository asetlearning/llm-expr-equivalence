import json
import re
from abc import ABC, abstractmethod

import pandas as pd

from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.equivalence_llm.prompts import result_patterns
from llm_expr_equivalence.schemas.common import Trichotomy


class LLMEvaluateResultsBase(ABC):
    def __init__(self, batch_file_name: str):
        self.batch_file_name = batch_file_name
        self.assess_data = self.__json_to_df__()

    @abstractmethod
    def __json_to_df__(self) -> pd.DataFrame:
        pass

    def evaluate(self, true_data: pd.DataFrame) -> float:
        """
        Evaluates results in assess_data against true_data
        """

        assess_data = self.__json_to_df__()
        print(assess_data)
        merged_df = (
            assess_data
            if assess_data.index.name == "exprs_hash"
            else assess_data.set_index("exprs_hash")
        ).join(
            (
                true_data
                if true_data.index.name == "exprs_hash"
                else true_data.set_index("exprs_hash")
            ),
            on="exprs_hash",
            how="left",
            lsuffix="_llm",
            rsuffix="_exprs",
            validate="one_to_one",
        )

        print(merged_df.to_string())
        print(merged_df[merged_df["equal"] != merged_df["result"]])
        return (merged_df["equal"] == merged_df["result"]).sum()

    @staticmethod
    def extract_result(content: str) -> Trichotomy | None:
        # see if converts to json as is
        try:
            result = json.loads(content)
            if "result" in result:
                return Trichotomy(result["result"])
        except Exception:
            pass

        for pattern in result_patterns:
            strip_out = content.replace("\n","").replace(" ", "")
            match = re.search(pattern, strip_out)
            if match:
                return Trichotomy(match.group(1))

        logger.info(f"Could not extract result from {content}")
        return None


class LLMEvaluateResultsOpenAI(LLMEvaluateResultsBase):
    def __init__(self, batch_file_name: str ):
        super().__init__(batch_file_name)

    def __json_to_df__(self) -> pd.DataFrame:
        """
        Convert batch processing results into a pandas DataFrame.
        Returns:
            DataFrame containing expression hashes and their comparison results
        Parses the output file and extracts results.
        """

        logger.info(f"Convert results from {self.batch_file_name} ...")
        rows = []
        with open(self.batch_file_name) as f:
            for line in f.readlines():
                llm_output = json.loads(line)

                result = LLMEvaluateResultsBase.extract_result(
                    llm_output["response"]["body"]["choices"][0]["message"]["content"]
                )
                rows.append((llm_output["custom_id"], result))
                llm_df = pd.DataFrame(rows, columns=["exprs_hash", "result"])

        return llm_df

class LLMEvaluateResultsGemini(LLMEvaluateResultsBase):

    def __json_to_df__(self) -> pd.DataFrame:
        """
        Convert batch processing results into a pandas DataFrame.
        Returns:
            DataFrame containing expression hashes and their comparison results
        Parses the output file and extracts results.
        """

        logger.info(f"Convert results from {self.batch_file_name} ...")
        rows = []
        with open(self.batch_file_name) as f:
            for line in f.readlines():
                llm_output = json.loads(line)
                result = LLMEvaluateResultsBase.extract_result(
                    llm_output["response"]["candidates"][0]["content"]["parts"][0]["text"]
                )
                exprs_hash = llm_output["request"]["labels"]["llm-equivalence-hash"]
                rows.append((exprs_hash, result))
                llm_df = pd.DataFrame(rows, columns=["exprs_hash", "result"])

        return llm_df
