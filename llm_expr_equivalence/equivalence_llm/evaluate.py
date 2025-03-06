import pandas as pd


def evaluate(assess_data: pd.DataFrame, true_data: pd.DataFrame) -> float:
    """
    Evaluates results in assess_data against true_data
    """

    merged_df = (
        assess_data
        if assess_data.index.name == "exprs_hash"
        else assess_data.set_index("exprs_hash")
    ).join(
        (
            true_data
            if true_data.index.name == "exprs_hash"
            else true_data.set_index("exprs_hash")
        ),
        on="exprs_hash",
        how="left",
        lsuffix="_llm",
        rsuffix="_exprs",
        validate="one_to_one",
    )

    print(merged_df[merged_df["equal"] != merged_df["result"]])
    return (merged_df["equal"] == merged_df["result"]).sum()
