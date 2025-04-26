from llm_expr_equivalence.equivalence_llm.equivalence_gemini import LLMEquivalenceGemini


def main():
    """
    Test features
    """
    # Data collections
    #    collect_and_store()

    # Compute equivalence
    #    compute_equivalences_sympy()
    #    compute_equivalences_mathserver()

    print("****", LLMEquivalenceGemini().equal("cos(pi*2)", "cos(pi + pi)"))

# Run the main function if the script is executed directly.
if __name__ == "__main__":
    main()
