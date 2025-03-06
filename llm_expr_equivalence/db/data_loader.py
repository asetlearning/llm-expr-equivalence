import pandas as pd

from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.db.mongo_tools import MongoConfig, mongo_db


class DataLoader:

    def __init__(self, **kwargs):
        db_cfg = MongoConfig(
            #        host="localhost",
            username="",
            password="",
            #        'mongo_ssl': False,
            database_name="llm-expr-equivalence",
        )

        with mongo_db(db_cfg) as db:
            math_df = pd.DataFrame(list(db.mathserver_results.find()))
            expressions_df = pd.DataFrame(list(db.expressions.find()))
        #            sympy_df = pd.DataFrame(list(db.sympy_results.find()))

        merged_df = math_df.set_index("exprs_hash").join(
            expressions_df.set_index("exprs_hash"),
            on="exprs_hash",
            how="left",
            lsuffix="_math",
            rsuffix="_exprs",
            validate="one_to_one",
        )

        self.load_df = merged_df
        self.limit: int | None = kwargs.get("limit", None)
        self.remove_trivial: bool = kwargs.get("remove_trivial", True)

        if self.limit > 0:
            self.load_df = self.load_df.head(self.limit)


    def data_recs(self):
        rec_count = 0
        for exprs_hash, row in self.load_df.iterrows():
            if self.limit is not None and rec_count >= self.limit:
                break

            if row["equal"] == "DONTKNOW":
                logger.error(f"DONTKNOW for {row}")
            elif self.remove_trivial and (row["expr1_exprs"]["is_const"] or row["expr2_exprs"]["is_const"]):
                logger.info(f"Skipping trivial: {row}")
            else:
                yield (
                    exprs_hash,
                    {
                        "ltx": row["expr1_math"]["ltx"],
                        "type": row["expr1_exprs"]["type"],
                        "is_const": row["expr1_exprs"]["is_const"],
                    },
                    {
                        "ltx": row["expr2_math"]["ltx"],
                        "type": row["expr2_exprs"]["type"],
                        "is_const": row["expr2_exprs"]["is_const"],
                    },
                    row["equal"],
                )
