from llm_expr_equivalence.data_collect.import_data import collect_and_store
from llm_expr_equivalence.common.grl2ltx import ParseGrlServer
from llm_expr_equivalence.equivalence_deterministic.equivalence_mathserver import (
    MathServer, compute_equivalences_mathserver
)
from llm_expr_equivalence.equivalence_deterministic.equivalence_sympy import (
    compute_equivalences_sympy,
    SympyMatching
)
from llm_expr_equivalence.equivalence_deterministic.equivalence_mathserver import compute_equivalences_mathserver

def main():
    """
    Test features
    """
    # Data collections
#    collect_and_store()

    # Compute equivalence
    compute_equivalences_sympy()
#    compute_equivalences_mathserver()

    # TMP
    # res = MathServer().get_types(["1","x"])
    # print(res)
    #
    # res = MathServer().are_of_type(["1","x"], "Constant")
    # print(res)

# Run the main function if the script is executed directly.
if __name__ == "__main__":
    main()


