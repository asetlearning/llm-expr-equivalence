from tqdm import tqdm

from llm_expr_equivalence.common.mathserver import MathServer
from llm_expr_equivalence.db.mongo_tools import (
    MongoConfig,
    create_index_if_not_exist,
    mongo_db,
)
from llm_expr_equivalence.schemas.common import Trichotomy

#  TODO: Optimize for distributed asynchronous calls


def compute_equivalences_mathserver(in_collection: str = "expressions_filtered"):
    db_cfg = MongoConfig(
        #        host="localhost",
        port=27019,
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name="llm-expr-equivalence",
    )

    mathserver = MathServer()
    with mongo_db(db_cfg) as db:
        create_index_if_not_exist(
            db, "mathserver_results", "exprs_hash_1", "exprs_hash", True
        )

        for q in tqdm(db[in_collection].find({})):
            expr1 = q["data"]["expr1"]["grl"]
            expr2 = q["data"]["expr2"]["grl"]

            result_equiv = mathserver.get_is_equivalent(expr1, expr2)
            result_match = Trichotomy.DONOTKNOW
            if result_equiv == Trichotomy.DONOTKNOW:
                result_match = (
                    Trichotomy.YES
                    if mathserver.get_does_match_reference(
                        expr1, expr2
                    )
                    else Trichotomy.NO
                )

            ltx1 = q["data"]["expr1"]["ltx"]
            ltx2 = q["data"]["expr2"]["ltx"]

            db.mathserver_results.insert_one(
                {
                    "exprs_hash": q["data"]["exprs_hash"],
                    "expr1": {
                        "grl": expr1,
                        "ltx": ltx1,
                    },
                    "expr2": {
                        "grl": expr2,
                        "ltx": ltx2,
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
