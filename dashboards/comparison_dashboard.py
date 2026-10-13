"""
LLM Expression Equivalence — Multi-File Comparison Dashboard

Compares experiment results across multiple JSONL result files from
Data/nebius/, Data/openai/, and Data/gemini/. For each selected file,
evaluates predictions against MongoDB ground truth and visualises:
  - Summary table: record count and accuracy per file
  - Grouped bar chart: correct vs incorrect by source, one group per file
  - Accuracy comparison bar chart across all selected experiments
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
    import itertools
    import sys
    from pathlib import Path

    import pandas as pd
    import plotly.graph_objects as go

    _project_root = Path(__file__).parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    DATA_DIR = _project_root / "Data"

    EXPERIMENT_PALETTE = [
        "#1976D2",  # blue
        "#388E3C",  # green
        "#F57C00",  # orange
        "#7B1FA2",  # purple
        "#00838F",  # teal
        "#C62828",  # red
        "#4527A0",  # deep purple
        "#2E7D32",  # dark green
    ]
    # Lighter/muted counterparts for incorrect bars (same hue family, desaturated)
    INCORRECT_PALETTE = [
        "#90CAF9",  # light blue
        "#A5D6A7",  # light green
        "#FFCC80",  # light orange
        "#CE93D8",  # light purple
        "#80DEEA",  # light teal
        "#EF9A9A",  # light red
        "#B39DDB",  # light deep purple
        "#C8E6C9",  # light dark green
    ]
    return DATA_DIR, EXPERIMENT_PALETTE, INCORRECT_PALETTE, Path, go, itertools, pd


@app.cell
def _discover_files(DATA_DIR, Path, mo):
    """
    Naming conventions (per project):
      nebius  : *-llm-result.jsonl
      openai  : *-llm-result.jsonl
      gemini  : gemini-*.jsonl
    """

    def _is_result_file(path: Path, provider: str) -> bool:
        if provider in ("nebius", "openai"):
            return path.name.endswith("-llm-result.jsonl")
        if provider == "gemini":
            return path.name.startswith("gemini-") and path.name.endswith(".jsonl")
        return False

    def _friendly_label(provider: str, filename: str) -> str:
        name = filename
        for suffix in ("-llm-result.jsonl", ".jsonl"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        for prefix in (f"{provider}-batch-", f"{provider}-"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return f"{provider} — {name}"

    result_files: dict[str, str] = {}
    for _provider in ("nebius", "openai", "gemini"):
        _folder = DATA_DIR / _provider
        if _folder.is_dir():
            for _f in sorted(_folder.iterdir()):
                if _is_result_file(_f, _provider):
                    _label = _friendly_label(_provider, _f.name)
                    result_files[_label] = str(_f)

    mo.stop(
        not result_files,
        mo.callout(mo.md("**No result JSONL files found in Data/.**"), kind="warn"),
    )
    return (result_files,)


@app.cell
def _file_selector(mo, result_files: dict[str, str]):
    file_multiselect = mo.ui.multiselect(
        options=result_files,
        value=list(result_files.keys())[:1],
        label="Select experiment files to compare",
    )
    file_multiselect
    return (file_multiselect,)


@app.cell
def _load_true_data():
    true_data = None
    db_error = None
    try:
        from llm_expr_equivalence.db.data_loader import DataLoader

        true_data = DataLoader(remove_trivial=True).load_df
    except Exception as _exc:
        db_error = str(_exc)
    return db_error, true_data


@app.cell
def _db_status(db_error, mo, true_data):
    (
        mo.callout(
            mo.md(
                f"**MongoDB unavailable** — evaluation requires a live DB.\n\n```\n{db_error}\n```"
            ),
            kind="danger",
        )
        if db_error
        else mo.callout(
            mo.md(f"Ground truth loaded: **{len(true_data):,}** records"),
            kind="success",
        )
    )
    return


@app.cell
def _evaluate_all(Path, file_multiselect, pd, true_data):
    """
    For each path selected in file_multiselect, infers the provider from the
    parent directory name, calls the matching LLMEvaluateResults* class, and
    merges all results into a single DataFrame tagged with 'file_label'.

    Exported:
        combined_df : pd.DataFrame | None
        summary     : list[dict]  — [{Experiment, Records, Accuracy, Accuracy (raw)}]
        eval_errors : list[str]
    """
    combined_df = None
    summary: list[dict] = []
    eval_errors: list[str] = []

    selected_paths: list[str] = file_multiselect.value if file_multiselect.value else []

    if true_data is None:
        eval_errors.append("MongoDB unavailable — cannot evaluate any file.")
    elif not selected_paths:
        eval_errors.append("No files selected.")
    else:
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

            frames: list[pd.DataFrame] = []

            for _path_str in selected_paths:
                _path = Path(_path_str)
                _provider = _path.parent.name
                _EvalClass = _PROVIDER_MAP.get(_provider)
                if _EvalClass is None:
                    eval_errors.append(
                        f"Unknown provider folder '{_provider}' for {_path.name}"
                    )
                    continue
                try:
                    _df, _acc = _EvalClass(_path_str).evaluate(true_data)

                    # Build a short friendly label from the file path
                    _stem = _path.name
                    for _sfx in ("-llm-result.jsonl", ".jsonl"):
                        if _stem.endswith(_sfx):
                            _stem = _stem[: -len(_sfx)]
                            break
                    for _pfx in (f"{_provider}-batch-", f"{_provider}-"):
                        if _stem.startswith(_pfx):
                            _stem = _stem[len(_pfx):]
                            break
                    _label = f"{_provider} — {_stem}"

                    _df = _df.copy()
                    _df["file_label"] = _label
                    _df["provider"] = _provider
                    frames.append(_df)
                    summary.append(
                        {
                            "Experiment": _label,
                            "Records": len(_df),
                            "Accuracy": f"{_acc:.1%}",
                            "Accuracy (raw)": _acc,
                        }
                    )
                except Exception as _file_exc:
                    eval_errors.append(f"{_path.name}: {_file_exc}")

            if frames:
                combined_df = pd.concat(frames, ignore_index=True)

        except Exception as _exc:
            eval_errors.append(str(_exc))
    return combined_df, eval_errors, summary


@app.cell
def _show_eval_errors(eval_errors: list[str], mo):
    mo.stop(not eval_errors, mo.md(""))
    mo.callout(
        mo.md(
            "**Evaluation errors:**\n\n" + "\n\n".join(f"- `{e}`" for e in eval_errors)
        ),
        kind="warn",
    )
    return


@app.cell
def _header(mo):
    mo.md("""
    ## Experiment Comparison Dashboard
    """)
    return


@app.cell
def _summary_table(mo, pd, summary: list[dict]):
    mo.stop(not summary, mo.md("_Select at least one file to see results._"))

    _summary_df = pd.DataFrame(
        [
            {
                "Experiment": r["Experiment"],
                "Records": r["Records"],
                "Accuracy": r["Accuracy"],
            }
            for r in summary
        ]
    )

    mo.vstack(
        [
            mo.md("### Summary — Accuracy per Experiment"),
            mo.ui.table(_summary_df),
        ]
    )
    return


@app.cell
def _accuracy_comparison_chart(
    EXPERIMENT_PALETTE,
    go,
    itertools,
    mo,
    summary: list[dict],
):
    mo.stop(not summary, mo.md(""))

    _labels = [r["Experiment"] for r in summary]
    _accs = [r["Accuracy (raw)"] for r in summary]
    _colors = list(itertools.islice(itertools.cycle(EXPERIMENT_PALETTE), len(_labels)))

    _fig_acc = go.Figure(
        go.Bar(
            x=_labels,
            y=_accs,
            marker_color=_colors,
            text=[f"{a:.1%}" for a in _accs],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<extra></extra>",
        )
    )
    _fig_acc.update_layout(
        title="Accuracy Comparison Across Experiments",
        title_font_color="#0D47A1",
        xaxis_title="Experiment",
        yaxis_title="Accuracy",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.1],
        template="plotly_white",
        paper_bgcolor="#FAFAFA",
        xaxis_tickangle=-35,
    )

    mo.vstack(
        [
            mo.md("### Accuracy Comparison"),
            mo.ui.plotly(_fig_acc),
        ]
    )
    return


@app.cell
def _outcomes_by_source(
    EXPERIMENT_PALETTE,
    INCORRECT_PALETTE,
    combined_df,
    go,
    itertools,
    mo,
    summary: list[dict],
):
    mo.stop(combined_df is None, mo.md("_No data loaded._"))

    _labels = [r["Experiment"] for r in summary]
    _correct_colors = list(itertools.islice(itertools.cycle(EXPERIMENT_PALETTE), len(_labels)))
    _incorrect_colors = list(itertools.islice(itertools.cycle(INCORRECT_PALETTE), len(_labels)))
    _correct_color: dict[str, str] = dict(zip(_labels, _correct_colors, strict=False))
    _incorrect_color: dict[str, str] = dict(zip(_labels, _incorrect_colors, strict=False))

    # Aggregate: file_label × source × outcome, then compute percentages per (file_label, source)
    _counts = (
        combined_df.groupby(["file_label", "source", "evaluate"])
        .size()
        .reset_index(name="count")
    )
    _totals = _counts.groupby(["file_label", "source"])["count"].transform("sum")
    _counts["pct"] = _counts["count"] / _totals * 100
    _counts["Outcome"] = _counts["evaluate"].map({True: "Correct", False: "Incorrect"})

    _fig_src = go.Figure()
    for _label in _labels:
        for _outcome, _bar_color in [
            ("Correct", _correct_color[_label]),
            ("Incorrect", _incorrect_color[_label]),
        ]:
            _sub = _counts[
                (_counts["file_label"] == _label) & (_counts["Outcome"] == _outcome)
            ]
            if _sub.empty:
                continue
            _fig_src.add_trace(
                go.Bar(
                    name=f"{_outcome}",
                    x=_sub["source"],
                    y=_sub["pct"],
                    marker_color=_bar_color,
                    legendgroup=_label,
                    legendgrouptitle_text=_label if _outcome == "Correct" else None,
                    hovertemplate=(
                        f"<b>{_label}</b><br>"
                        "Source: %{x}<br>"
                        f"Outcome: {_outcome}<br>"
                        "Share: %{y:.1f}%<extra></extra>"
                    ),
                )
            )

    _fig_src.update_layout(
        title="Correct vs Incorrect Outcomes by Source (%)",
        title_font_color="#0D47A1",
        barmode="group",
        xaxis_title="Source",
        yaxis_title="Share (%)",
        yaxis_ticksuffix="%",
        yaxis_range=[0, 105],
        template="plotly_white",
        paper_bgcolor="#FAFAFA",
        legend={"groupclick": "toggleitem", "tracegroupgap": 10},
        xaxis_tickangle=-35,
    )

    mo.vstack(
        [
            mo.md("### Outcomes by Source"),
            mo.ui.plotly(_fig_src),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
