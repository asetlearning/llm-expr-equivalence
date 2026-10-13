"""
LLM Expression Equivalence Results Dashboard

Reads local JSONL result files from Data/nebius/, Data/gemini/, Data/openai/,
evaluates predictions against ground truth from MongoDB, and visualises:
  - Overall accuracy stat
  - Correct/incorrect proportion by source (bar chart)
  - Accuracy heatmap by expr1_type x expr2_type
  - Classification report table
"""

# Created by: Alex Myasnikov
# Date: March 3, 2026

import marimo

__generated_with = "0.20.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _imports():
    import sys
    from pathlib import Path

    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    DATA_DIR = _project_root / "Data"
    COLOR_CORRECT = "#1976D2"
    COLOR_INCORRECT = "#EF5350"
    return COLOR_CORRECT, COLOR_INCORRECT, DATA_DIR, Path, go, pd, px


@app.cell
def _discover_files(DATA_DIR, Path, mo):
    def _is_result_file(path: Path, provider: str) -> bool:
        if provider in ("nebius", "openai"):
            return path.name.endswith("-llm-result.jsonl")
        if provider == "gemini":
            return path.name.startswith("gemini-") and path.name.endswith(".jsonl")
        return False

    result_files: dict[str, str] = {}
    for _provider in ("nebius", "openai", "gemini"):
        _folder = DATA_DIR / _provider
        if _folder.is_dir():
            for _f in sorted(_folder.iterdir()):
                if _is_result_file(_f, _provider):
                    result_files[f"{_provider}/{_f.name}"] = str(_f)

    mo.stop(
        not result_files,
        mo.callout(mo.md("**No result JSONL files found in Data/.**"), kind="warn"),
    )

    file_dropdown = mo.ui.dropdown(
        options=result_files,
        value=next(iter(result_files)),
        label="Select result file",
    )
    return (file_dropdown,)


@app.cell
def _show_dropdown(file_dropdown):
    file_dropdown


@app.cell
def _load_true_data():
    try:
        from llm_expr_equivalence.db.data_loader import DataLoader
        true_data = DataLoader(remove_trivial=True).load_df
        db_error = None
    except Exception as _exc:
        true_data = None
        db_error = str(_exc)
    return db_error, true_data


@app.cell
def _db_status(db_error, mo, true_data):
    _out = (
        mo.callout(mo.md(f"**MongoDB unavailable.**\n\n```\n{db_error}\n```"), kind="danger")
        if db_error
        else mo.callout(mo.md(f"Ground-truth loaded: **{len(true_data):,}** records"), kind="success")
    )
    return


@app.cell
def _evaluate_results(Path, file_dropdown, true_data):
    results_data = None
    accuracy = None
    eval_error = None
    selected_path = ""

    try:
        from llm_expr_equivalence.equivalence_llm.evaluate import (
            LLMEvaluateResultsGemini,
            LLMEvaluateResultsNebius,
            LLMEvaluateResultsOpenAI,
        )

        _PROVIDER_MAP = {
            "nebius": LLMEvaluateResultsNebius,
            "openai": LLMEvaluateResultsOpenAI,
            "gemini": LLMEvaluateResultsGemini,
        }

        selected_path = file_dropdown.value

        if true_data is None:
            eval_error = "MongoDB unavailable — cannot evaluate."
        else:
            _provider = Path(selected_path).parent.name
            _EvalClass = _PROVIDER_MAP.get(_provider)
            if _EvalClass is None:
                eval_error = f"Unknown provider folder: `{_provider}`"
            else:
                results_data, accuracy = _EvalClass(selected_path).evaluate(true_data)
    except Exception as _exc:
        if eval_error is None:
            eval_error = str(_exc)
    return accuracy, eval_error, results_data


@app.cell
def _eval_status(accuracy, eval_error, mo, results_data):
    (
        mo.callout(mo.md(f"**Evaluation failed.**\n\n```\n{eval_error}\n```"), kind="danger")
        if eval_error
        else mo.callout(
            mo.md(f"Evaluated **{len(results_data):,}** records — accuracy **{accuracy:.1%}**"),
            kind="success",
        )
        if results_data is not None
        else mo.md("_Waiting..._")
    )


@app.cell
def _accuracy_stat(accuracy, mo):
    mo.stop(accuracy is None, mo.md("_No data._"))
    mo.stat(
        label="Overall Accuracy",
        value=f"{accuracy:.1%}",
        caption="proportion of correct predictions",
    )


@app.cell
def _source_bar_chart(COLOR_CORRECT, COLOR_INCORRECT, mo, px, results_data):
    mo.stop(results_data is None, mo.md("_No data._"))

    _counts = (
        results_data.groupby(["source", "evaluate"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={True: "Correct", False: "Incorrect"})
    )
    for _col in ("Correct", "Incorrect"):
        if _col not in _counts.columns:
            _counts[_col] = 0

    _proportions = _counts.div(_counts.sum(axis=1), axis=0).reset_index()
    _long = _proportions.melt(id_vars="source", var_name="Outcome", value_name="Proportion")

    _fig = px.bar(
        _long,
        x="source",
        y="Proportion",
        color="Outcome",
        barmode="stack",
        color_discrete_map={"Correct": COLOR_CORRECT, "Incorrect": COLOR_INCORRECT},
        title="Evaluation Outcome Proportion by Source",
        labels={"source": "Source", "Proportion": "Proportion"},
        template="plotly_white",
    )
    _fig.update_layout(
        title_font_color="#0D47A1",
        legend_title_text="Outcome",
        xaxis_tickangle=-35,
        yaxis_tickformat=".0%",
        paper_bgcolor="#FAFAFA",
    )
    return


@app.cell
def _heatmap_chart(go, mo, results_data):
    mo.stop(results_data is None, mo.md("_No data._"))

    _pivot = results_data.pivot_table(
        index="expr1_type",
        columns="expr2_type",
        values="evaluate",
        aggfunc="mean",
    )
    _fig2 = go.Figure(
        data=go.Heatmap(
            z=_pivot.values,
            x=list(_pivot.columns),
            y=list(_pivot.index),
            colorscale=[[0.0, "#E3F2FD"], [0.5, "#1976D2"], [1.0, "#0D47A1"]],
            zmin=0,
            zmax=1,
            colorbar={"title": "Accuracy"},
            hovertemplate="expr1: %{y}<br>expr2: %{x}<br>accuracy: %{z:.1%}<extra></extra>",
        )
    )
    _all_types = sorted(set(list(_pivot.columns) + list(_pivot.index)))
    _fig2.update_layout(
        title="Accuracy by Expression Type Pair",
        title_font_color="#0D47A1",
        xaxis={"title": "expr2_type", "categoryorder": "array", "categoryarray": _all_types, "tickangle": -45},
        yaxis={
            "title": "expr1_type",
            "categoryorder": "array",
            "categoryarray": _all_types,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        width=600,
        height=600,
        template="plotly_white",
        paper_bgcolor="#FAFAFA",
    )
    mo.ui.plotly(_fig2)


@app.cell
def _accuracy_by_pair(mo, results_data):
    mo.stop(results_data is None, mo.md("_No data._"))

    _stats = (
        results_data.groupby(["expr1_type", "expr2_type"])["evaluate"]
        .agg(accuracy="mean", total="count")
        .sort_values("accuracy", ascending=False)
        .reset_index()
    )
    mo.vstack([mo.md("### Accuracy by Expression Type Pair"), mo.ui.table(_stats)])


@app.cell
def _classification_report(mo, pd, results_data):
    from sklearn.metrics import classification_report

    mo.stop(results_data is None, mo.md("_No data._"))

    _y_true = results_data["equal"].astype(str)
    _y_pred = results_data["equal_llm"].astype(str)

    cr = classification_report(_y_true, _y_pred, zero_division=0)
    mo.vstack([mo.md("### Classification Report"), mo.md(f"```\n{cr}```\n")])

@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
