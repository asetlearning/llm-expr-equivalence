import argparse
import json

import pandas as pd

from llm_expr_equivalence.config.settings import logger
from llm_expr_equivalence.db.mongo_tools import MongoConfig, mongo_db


class DataLoader:

    def __init__(self, **kwargs):
        db_cfg = MongoConfig(
            #        host="localhost",
            port=27019,
            username="",
            password="",
            #        'mongo_ssl': False,
            database_name="llm-expr-equivalence",
        )

        with mongo_db(db_cfg) as db:
            # Ignore the ones that rewrite DONTKNOW or match is NO. This is most likely junk
            query = {
                "$or": [
                    {"result_equiv": {"$ne": "DONTKNOW"}},
                    {"result_match": {"$ne": "NO"}},
                ]
            }

            math_df = pd.DataFrame(list(db.mathserver_results.find(query)))
            expressions_df = pd.DataFrame(
                [r["data"] for r in list(db.expressions_filtered.find())]
            )
        #            sympy_df = pd.DataFrame(list(db.sympy_results.find()))

        merged_df = math_df.set_index("exprs_hash").join(
            expressions_df.set_index("exprs_hash"),
            on="exprs_hash",
            how="left",
            lsuffix="_math",
            rsuffix="_exprs",
            validate="one_to_one",
        )

        bad_case = "3599875f77715a9d8de4dc7c87b3dce4387bdf5e00526876a705f0c0edb1a3b1"

        self.load_df = merged_df[merged_df.index != bad_case]
        self.limit: int | None = kwargs.get("limit", None)
        self.remove_trivial: bool = kwargs.get("remove_trivial", True)

        if self.limit and self.limit > 0:
            self.load_df = self.load_df.head(self.limit)

    def data_recs(self):
        rec_count = 0
        for exprs_hash, row in self.load_df.iterrows():
            if self.limit is not None and rec_count >= self.limit:
                break

            if row["equal"] == "DONTKNOW":
                logger.error(f"DONTKNOW for {row}")
            elif self.remove_trivial and (
                row["expr1_exprs"]["is_const"] or row["expr2_exprs"]["is_const"]
            ):
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
                    row,
                )

    def export_to_jsonl(self, filepath: str):
        with open(filepath, "w") as f:
            for exprs_hash, expr1, expr2, equal, row in self.data_recs():
                record = {
                    "exprs_hash": exprs_hash,
                    "expr1": expr1,
                    "expr2": expr2,
                    "equal": equal,
                    "details": row.drop(["_id_math","_id_exprs"], errors="ignore").to_dict(),
                }
                f.write(json.dumps(record) + "\n")
        logger.info(f"Exported data to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Export data to JSONL file")
    parser.add_argument("filename", type=str, help="Output JSONL filename")
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit number of records"
    )
    parser.add_argument(
        "--include-trivial", action="store_true", help="Include trivial expressions"
    )

    args = parser.parse_args()

    loader = DataLoader(limit=args.limit, remove_trivial=not args.include_trivial)
    loader.export_to_jsonl(args.filename)


if __name__ == "__main__":
    main()
