
from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.db.data_loader import DataLoader
from llm_expr_equivalence.equivalence_llm.evaluate import LLMEvaluateResultsGemini


def main():
    """
    Test features
    """

    dataloader = DataLoader(limit=10, remove_trivial=True)

    # llm = LLMEquivalenceBatchGemini("gemini-2.0-flash-lite-001")
    # llm.submit_batch(dataloader)
    # llm.get_batch_result()
    #llm.get_batch_result()
#    llm_df = llm.get_results_df("openai-batch-20250305-154500-llm-result.jsonl")
#    logger.info(f"Evaluation result: {evaluate(llm_df, dataloader.load_df)}")
    evaluate = LLMEvaluateResultsGemini("Data/gemini/gemini-2025-04-25T21_24_55_predictions.jsonl")
    logger.info(f"Evaluation result: {evaluate.evaluate(dataloader.load_df)}")


# Run the main function if the script is executed directly.
if __name__ == "__main__":
    main()
