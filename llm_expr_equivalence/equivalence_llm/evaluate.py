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

    def evaluate(self, true_data: pd.DataFrame) -> tuple[pd.DataFrame, float]:
        """
        Evaluates results in assess_data against true_data
        """

        assess_data = self.__json_to_df__()
        logger.info(f"Assessment data:\n{assess_data}")
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
            rsuffix="_golden",
            validate="one_to_one",
        )
        merged_df["evaluate"] = merged_df["equal"] == merged_df["equal_llm"]
        # logger.info(f"Merged data:\n{merged_df.to_string()}")
        # logger.info(f"Mismatches:\n{merged_df[merged_df['equal'] != merged_df['result']]}")


        # add some postprocessing
        # FIXME: this is just a bad record hack
        merged_df = (merged_df[merged_df.index != "3599875f77715a9d8de4dc7c87b3dce4387bdf5e00526876a705f0c0edb1a3b1"])
        # add types as separate columns
        merged_df["expr1_type"] = merged_df["expr1_exprs"].apply(lambda x: x["type"])
        merged_df["expr2_type"] = merged_df["expr2_exprs"].apply(lambda x: x["type"])

        return merged_df, merged_df["evaluate"].sum()/(len(merged_df))

    @staticmethod
    def extract_result(content: str | None) -> Trichotomy | None:
        if not content:
            logger.error("Empty content")
            return None
        # see if converts to json as is
        try:
            result = json.loads(content)
            if "result" in result:
                return Trichotomy(result["result"])
        except Exception:
            pass

        if content:
            for pattern in result_patterns:
                strip_out = content.replace("\n","").replace(" ", "")
                match = re.search(pattern, strip_out)
                if match:
                    try:
                       return Trichotomy(match.group(1))
                    except ValueError as e:
                        logger.error(f"Could not extract result from {content[:100]}: {e}")
                        return Trichotomy.DONOTKNOW

            logger.error(f"Could not extract result from {content[:100]}")
        else:
            logger.error("Empty content")
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

class LLMEvaluateResultsNebius(LLMEvaluateResultsBase):
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

                result_content = llm_output["response"]["choices"][0]["message"]["content"]
                result: Trichotomy | None
                if result_content is None or result_content == "":
#                    logger.error(f"Empty result in {llm_output}")
                    result = Trichotomy.DONOTKNOW
                else:
                    result = LLMEvaluateResultsBase.extract_result(result_content)

                rows.append((llm_output["custom_id"], result))

            llm_df = pd.DataFrame(rows, columns=["exprs_hash", "equal_llm"])

        return llm_df
