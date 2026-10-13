
from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.db.data_loader import DataLoader
from llm_expr_equivalence.equivalence_llm.evaluate import LLMEvaluateResultsOpenAI


def main():
    """
    Test features
    """

    dataloader = DataLoader(limit=10, remove_trivial=True)
    ## llm = LLMEquivalenceOpenAI("gpt-4o-mini-2024-07-18")
    #llm.submit_batch(dataloader)
    #llm.get_batch_result()


    evaluate = LLMEvaluateResultsOpenAI("Data/openai/openai-batch-20250305-154500-llm-result.jsonl")
    logger.info(f"Evaluation result: {evaluate.evaluate(dataloader.load_df)}")

# Run the main function if the script is executed directly.
if __name__ == "__main__":
    main()
