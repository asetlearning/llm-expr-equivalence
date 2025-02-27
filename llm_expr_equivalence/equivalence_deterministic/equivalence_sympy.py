import sympy
from latex2sympy2 import latex2sympy, latex2latex

from llm_expr_equivalence.settings import logger
from llm_expr_equivalence.schemas.common import Trichotomy
from llm_expr_equivalence.db.mongo_tools import (
    mongo_db,
    MongoConfig,
    create_index_if_not_exist,
)

#  TODO: Optimize for distributed asynchronous calls


class SympyMatching:

    @staticmethod
    def parse(expr: str) -> str | None:
        try:
            a = latex2sympy(expr)

            if isinstance(a, list) and len(a) == 1:
                a = a[0]

            logger.info(f"Parse: {expr} -> {a}")
            return a
        except Exception as e:
            logger.error(f" ------ Sympy parser failed for {expr}: {str(e)}")
            return None

    @staticmethod
    def equal(expr1: str, expr2: str) -> Trichotomy:
        a = SympyMatching.parse(expr1)
        b = SympyMatching.parse(expr2)

        if a is None or b is None:
            return Trichotomy.DONOTKNOW

        if isinstance(a, sympy.Eq) and isinstance(b, sympy.Eq):
            a = a.lhs - a.rhs
            b = b.lhs - b.rhs

        try:
            return Trichotomy.YES if a.equals(b) else Trichotomy.NO
        except Exception as e:
            logger.error(f"Sympy equivalence failed for {a} {b}: {str(e)}")
            return Trichotomy.DONOTKNOW


def compute_equivalences_sympy():
    db_cfg = MongoConfig(
        #        host="localhost",
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name="llm-expr-equivalence",
    )

    with mongo_db(db_cfg) as db:
        create_index_if_not_exist(
            db, "sympy_results", "exprs_hash_1", "exprs_hash", True
        )

        for q in db.expressions.find({}):
            result = SympyMatching.equal(q["expr1"]["ltx"], q["expr2"]["ltx"])
            db.sympy_results.insert_one(
                {
                    "exprs_hash": q["exprs_hash"],
                    "expr1": {
                        "grl": q["expr1"]["grl"],
                        "ltx": q["expr1"]["ltx"],
                    },
                    "expr2": {
                        "grl": q["expr2"]["grl"],
                        "ltx": q["expr2"]["ltx"],
                    },
                    "equal": result,
                }
            )
