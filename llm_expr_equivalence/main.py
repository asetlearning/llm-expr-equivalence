# from llm_expr_equivalence.equivalence_llm.equivalence_gemini import LLMEquivalenceGemini
from llm_expr_equivalence.equivalence_deterministic.equivalence_mathserver import compute_equivalences_mathserver


def main():
    """
    Test features
    """
    # Data collections
#    collect_and_store()

    # assign types and parse grl of expressions in mongodb
    # label_expressions()

    # Apply filters to extract expressions for dataset
    # apply_filters(max_group_sample_size=1000)

    # Compute equivalence
    #    compute_equivalences_sympy()
    compute_equivalences_mathserver()

    # Models:
    # deepseek/deepseek-prover-v2:free
    # qwen/qwen-2.5-72b-instruct:free

    # Examples:
    # 3 x-4 y=12\n  2. -4 y=12\n
    # expr1 = "3 x-4 y=12"
    # expr2 = "-4 y=12"
    # expr1 = "\\frac{1}{21}\\left(\\frac{1}{2}\\left(\\cot\\left(10x\\right)\\right)^4+\\frac{1}{3}"\
    #         "\\left(\\cot\\left(10x\\right)\\right)^6\\right)+C"
    #
    # expr2 = "-\\frac{3\\sin\\left(10x\\right)^2-2}{126\\sin\\left(10x\\right)^6}+C"
    # print(
    #     "****",
    #     LLMEquivalenceOpenAPI("deepseek/deepseek-r1-0528:free").equal(expr1, expr2),
    # )


# Run the main function if the script is executed directly.
if __name__ == "__main__":
    main()
