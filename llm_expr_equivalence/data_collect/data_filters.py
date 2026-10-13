from math import log, tanh

import pandas as pd
from tqdm import tqdm

from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.db.mongo_tools import MongoConfig, mongo_db

TYPE_RANK = {
    # 0: Unknown / undefined
    "Unknown": 0,
    "Undefined": 0,
    # 1: Atomic
    "Integer": 1,
    "Rational": 1,
    "Constant": 1,
    "ConstantBucket": 1,
    "Variable": 1,
    "Point": 1,
    # 2: Basic algebraic (mostly univariate / linear)
    "Polynomial": 2,
    "UnivariatePolynomial": 2,
    "LinPolynomial": 2,
    "Line": 2,
    "UnivariateLinPolynomial": 2,
    # 3: Set/interval constructs + 2D regions
    "Interval": 3,
    "IntervalCombination": 3,
    "Set": 3,
    "PSet": 3,
    "Region2D": 3,
    # 4: Equation is typically one-variable equation
    "Equation": 4,
    "Inequality": 4,
    "Function": 4,
    "UnivariateFunction": 4,
    # 5: Higher algebraic / combinatorial series-seq basics
    "MultivariatePolynomial": 5,
    "MultivariateLinPolynomial": 5,
    "PowerSeries": 5,
    "NumSeries": 5,
    "Sequence": 5,
    "GeneralTermSeq": 5,
    # 6: Rational / quotient (univariate-leaning)
    "Quotient": 6,
    "RationalFunction": 6,
    "UnivariateRationalFunction": 6,
    "QuotientFunction": 6,
    "UnivariateQuotientFunction": 6,
    # 7: Multivariate rational + basic function + vectors
    "MultivariateRationalFunction": 7,
    "MultivariateQuotientFunction": 7,
    "ExplicitVector": 7,
    "VectorExpression": 7,
    # 8: Multivariate functions + general series + projections
    "MultivariateFunction": 8,
    "Series": 8,
    "GeneralTermSeries": 8,
    "VectorProjection": 8,
    "ScalarProjection": 8,
    # 9: Intro calculus / limits & directional/partial
    "Limit": 9,
    "UnivariateLimit": 9,
    "VectorLimit": 9,
    "DirectionalDerivative": 9,
    "PartialDerivative": 9,
    # 10: Heavier multivariate calculus operators
    "MultivariateLimit": 10,
    "Derivative": 10,
    "Gradient": 10,
    "VectorDerivative": 10,
    # 11: Integrals (incl. multiple / vector)
    "Integral": 11,
    "DoubleIntegral": 11,
    "TripleIntegral": 11,
    "VectorIntegral": 11,
    "MathTerm": 11,
    # 12: 3D regions + list container
    "Region3D": 12,
    "List": 12,
    # 12: Relational / logical statements
    "Equality": 13,
    "Predicate": 13,
    # 13: Systems
    "System": 14,
    # 14: Meta / plotting semantics
    "Plottable": 15,
}


def get_pair_type_rank(t1: str, t2: str) -> int:
    return max(TYPE_RANK[t1], TYPE_RANK[t2]) + abs(TYPE_RANK[t1] - TYPE_RANK[t2])


def get_expr_semantic_complexity(expr: str) -> int:
    # Assuming that complexity depends on tree branching. '(' occurs in any operators or function calls in GRL.
    # This will not distinguish between parenthesis and limit(..) but type rank will take caro of that.
    # Think if should add "=", ","
    branching_symbols = ["(", "+", "/", "<", "-", ">"]
    return sum(expr.count(s) for s in branching_symbols)


def get_pair_lex_rank(e1, t1, e2, t2) -> tuple[int, int]:
    return get_pair_type_rank(t1, t2), get_expr_semantic_complexity(
        e1
    ) + get_expr_semantic_complexity(e2)


def get_pair_score(e1, t1, e2, t2) -> float:
    pair_len = get_expr_semantic_complexity(e1) + get_expr_semantic_complexity(e2)
    return get_pair_type_rank(t1, t2) + 0.2 * tanh(log(1 + pair_len) / 3.0)


def apply_filters(
    in_collection: str = "expressions",
    out_collection: str = "expressions_filtered",
    max_group_sample_size: int = 100,
):
    db_cfg = MongoConfig(
        #        host="localhost",
        port=27019,
        username="",
        password="",
        #        'mongo_ssl': False,
        database_name="llm-expr-equivalence",
    )

    logger.info(
        f"Filtering {in_collection} to {out_collection}. Max group sample size: {max_group_sample_size}"
    )

    query = {"$and": [{"expr1.type": {"$ne": None}}, {"expr2.type": {"$ne": None}}]}

    all_expressions = []
    with mongo_db(db_cfg) as db:
        for d in tqdm(db[in_collection].find(query), desc="Processing expressions"):
            if d["expr1"]["ltx"] is not None and d["expr2"]["ltx"] is not None:
                lex_score = get_pair_lex_rank(
                    d["expr1"]["ltx"],
                    d["expr1"]["type"],
                    d["expr2"]["ltx"],
                    d["expr2"]["type"],
                )
                all_expressions.append(
                    {
                        "data": d,
                        "pair_type_rank": lex_score[0],
                        "pair_len": lex_score[1],
                    }
                )

        #    all_expressions = sorted(all_expressions, key=lambda x: x["score"], reverse=True)
        # print("\n".join([str(e) for e in all_expressions[:10]]))
        df = pd.DataFrame(all_expressions)
        df_sampled = (
            df[df["pair_type_rank"] > 1]
            .groupby("pair_type_rank")
            .apply(lambda x: x.sample(n=min(len(x), max_group_sample_size)))
            .reset_index(drop=True)
        )

        db[out_collection].insert_many(df_sampled.to_dict(orient="records"))

    return
