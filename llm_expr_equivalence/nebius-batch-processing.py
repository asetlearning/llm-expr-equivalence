
from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.db.data_loader import DataLoader
from llm_expr_equivalence.equivalence_llm.evaluate import LLMEvaluateResultsNebius


def main():
    """
    Test features
    """

    dataloader = DataLoader(remove_trivial=True) #, limit=10
    # llm = LLMEquivalenceNebius("deepseek-ai/DeepSeek-R1-0528")
    # llm = LLMEquivalenceNebius("openai/gpt-oss-20b")
    # llm.submit_batch(dataloader)
    # llm.get_batch_result()


    evaluate = LLMEvaluateResultsNebius(
        "Data/nebius/nebius-batch-gpt-oss-20b-text-high-10000tkns-20260202-163332-llm-result.jsonl"
    )
    merged_df, result = evaluate.evaluate(dataloader.load_df)
    logger.info(f"Evaluation result: {result}")

# Run the main function if the script is executed directly.
if __name__ == "__main__":
    main()
