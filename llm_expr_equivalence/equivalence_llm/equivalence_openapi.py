
from openai import OpenAI

from llm_expr_equivalence.config.settings import get_settings, logger
from llm_expr_equivalence.equivalence_llm.equivalence_base import LLMEquivalenceBase
from llm_expr_equivalence.equivalence_llm.evaluate import LLMEvaluateResultsBase
from llm_expr_equivalence.equivalence_llm.prompts import equivalence_prompt
from llm_expr_equivalence.schemas.common import Trichotomy


class LLMEquivalenceOpenAPI(LLMEquivalenceBase):
    """
    OpenAPI implementation for mathematical expression equivalence checking.

    Supports single comparisons using OpenAPI format.
    Should work for any services supporting OpenAPI completion.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self.client = OpenAI(
            api_key=get_settings().openai_api_key.get_secret_value(),
            base_url=get_settings().openai_api_api_base_url,
            timeout=300,
        )

    def __equal__(self, expr1, expr2) -> Trichotomy:
        """
        Compare two expressions for equivalence using OpenAPI chat completion.
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
        reasoning = getattr(completion.choices[0].message, "reasoning", None)
        if reasoning:
            logger.info(f"Reasoning: {reasoning}")

        result = LLMEvaluateResultsBase.extract_result(completion.choices[0].message.content)
        if result is None:
            raise Exception("Failed to extract result from OpenAPI completion")

        return result
