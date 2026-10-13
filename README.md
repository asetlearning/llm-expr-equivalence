# llm-expr-equivalence

Benchmark of how well LLMs decide whether two mathematical expressions are
equivalent, measured against deterministic ground truth (MathServer, SymPy).

## Goals

* Build a large, deduplicated dataset of real expression pairs (sourced from
  MathServer assist logs) sampled across types and complexity levels.
* Produce ground-truth equivalence labels with deterministic engines.
* Ask LLMs the same question (`YES` / `NO` / `DONTKNOW`) via single calls or
  provider batch APIs, and compare accuracy across models and providers.
* Analyze results (accuracy, confusion matrices, per-bucket breakdowns) via
  marimo dashboards, notebooks and markdown reports.
* Export the dataset as JSONL for sharing / model training.

## Pipeline

1. **Collect** — `data_collect/import_data.py` scans the source MongoDB,
   dedups pairs (symmetric `exprs_hash`), stores in `expressions`; second pass
   labels types and LaTeX.
2. **Filter** — `data_collect/data_filters.py` samples up to N pairs per
   (type rank, length) group into `expressions_filtered`.
3. **Ground truth** — `equivalence_deterministic/` (MathServer, SymPy) writes
   `*_results` collections.
4. **LLM evaluation** — `equivalence_llm/` submits pairs to OpenAI, Gemini
   (Vertex AI), Nebius or any OpenAI-compatible API (e.g. OpenRouter).
5. **Analysis** — `equivalence_llm/evaluate.py`, `process_results.py`,
   `dashboards/`, `Notebooks/`.

## Tools

* Python 3.12, Poetry, poethepoet
* MongoDB (PyMongo), pandas
* OpenAI SDK, LangChain, google-genai + Google Cloud Storage
* SymPy, latex2sympy2, MathServer client
* FastAPI + Hypercorn (HTTP API)
* marimo, plotly, scikit-learn, Jupyter (analysis)
* ruff, mypy, pytest, pre-commit

## Dev Prerequisites

Install the following:

* Python 3.12
* [Poetry](https://python-poetry.org/docs/#installation)

Run the following commands:

```bash
poetry install
poetry run pre-commit install
cp llm_expr_equivalence/.env.example .env
```

Populate the `.env` file (repo root) with required values: provider API keys
(`OPENAI_API_KEY`, `NEBIUS_API_KEY`, `GEMINI_API_KEY`), `MATHSERVER_URL`,
`PARSER_URL`, Google Cloud project/bucket settings. Run commands from the repo
root so `.env` is found.

## Usage

```bash
poetry run poe test        # tests
poetry run poe typecheck   # mypy
poetry run poe lint        # ruff --fix
poetry run poe dev         # dev API server
```

Batch runs per provider (toggle submit vs. evaluate in each script's `main()`):

```bash
poetry run python -m llm_expr_equivalence.open-ai-batch-processing
poetry run python -m llm_expr_equivalence.gemini-batch-processing
poetry run python -m llm_expr_equivalence.nebius-batch-processing
```

Results land in `Data/<provider>/`.

Dashboards:

```bash
poetry run marimo edit dashboards/dashboard.py
poetry run marimo edit dashboards/comparison_dashboard.py
```

# Reports

Reports are mostly run from jupyter notebooks

Use

```bash
/opt/anaconda3/bin/jupyter nbconvert compare_results.ipynb \
  --to markdown \
  --no-input

```

To extract outputs into a markdown file.

# Processing prediction JSONL files

For the flat prediction JSONL schema (one record per prediction with fields
`uuid`, `expr1_ltx`, `expr2_ltx`, `ground_truth`, `completion`, `prediction`,
`correct`), use:

```bash
poetry run python -m llm_expr_equivalence.process_results <input.jsonl> \
    [--split-prefix PREFIX] [--original-data <orig.jsonl>]
```

Re-extracts `prediction` from `completion` when null (handles `{"result": ...}`
JSON and `\boxed{...}` / `\boxed{\text{...}}` notations, e.g. qwen2.5 Math),
prints accuracy, confusion matrix, and per-bucket YES/NO breakdown as
markdown to stdout. The input file is never modified.

With `--split-prefix PREFIX`, writes `PREFIX.correct.jsonl` and
`PREFIX.incorrect.jsonl` of the prediction records.

With `--original-data PATH` (requires `--split-prefix`), additionally splits
the original dataset JSONL into `PREFIX.original.correct.jsonl` and
`PREFIX.original.incorrect.jsonl`, matching each original record's
`exprs_hash` to the prediction `uuid`. The count of unmatched original
records is reported in stdout.
