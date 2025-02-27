from llm_expr_equivalence.common.mathserver import MathServer
from llm_expr_equivalence.schemas.common import Trichotomy
from llm_expr_equivalence.db.mongo_tools import (
    mongo_db,
    MongoConfig,
    create_index_if_not_exist,
)

#  TODO: Optimize for distributed asynchronous calls


def compute_equivalences_mathserver():
    out_cfg = MongoConfig(
        #        host="localhost",
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name="llm-expr-equivalence",
    )

    mathserver = MathServer()
    with mongo_db(out_cfg) as db:
        create_index_if_not_exist(
            db, "mathserver_results", "exprs_hash_1", "exprs_hash", True
        )

        for q in db.expressions.find({}):
            result_equiv = mathserver.get_is_equivalent(q["expr1"]["grl"], q["expr2"]["grl"])
            result_match = Trichotomy.DONOTKNOW
            if result_equiv == Trichotomy.DONOTKNOW:
                result_match = (
                    "YES"
                    if mathserver.get_does_match_reference(
                        q["expr1"]["grl"], q["expr2"]["grl"]
                    )
                    else "NO"
                )

            db.mathserver_results.insert_one(
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
                    "equal": (
                        result_equiv
                        if result_equiv != Trichotomy.DONOTKNOW
                        else result_match
                    ),
                    "result_equiv": result_equiv,
                    "result_match": result_match,
                }
            )
