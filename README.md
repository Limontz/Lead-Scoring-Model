# Assignment Jet HR

Contact: `marcosimonte96@gmail.com`

Take-home project for:
- funnel analysis (volumes, conversions, temporal trends)
- a first lead scoring model

Task reports:
- Task 1 (Funnel Analysis): `Task1_Funnel_Analysis.pdf`
- Task 2 (Lead Scoring Model): `Task2_Lead_Scoring_Model.pdf`

## Quickstart (fresh clone)

```bash
git clone git@github.com:Limontz/Assignment-Jet-HR.git
cd Assignment-Jet-HR
uv sync --frozen
uv run pytest -q
uv run python main.py
```

If `uv sync --frozen` fails because a compatible Python is not available, run:

```bash
uv python install 3.13
uv sync --frozen
```

If all commands pass, the project is set up correctly.

## Requirements

- Python `>=3.13` (from `pyproject.toml`)
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency management

## 1) Install `uv`

### macOS (recommended)

```bash
brew install uv
```

### Alternative (official installer)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
```

## 2) Install dependencies

From the project root:

```bash
uv sync --frozen
```

Use `--frozen` to reproduce the exact locked environment from `uv.lock`.

## 3) Marimo notebooks

This repo stores notebooks as Marimo apps (`.py` files) in `notebooks/`.
Run them through `uv` so they use the project environment.

Open a notebook in Marimo editor:

```bash
uv run marimo edit notebooks/Exploratory_analysis.py
uv run marimo edit notebooks/Static_Funnel_Analysis.py
uv run marimo edit notebooks/Temporal_funnel_analysis.py
```

## 4) Run analysis in notebooks

Main analysis notebooks:
- `notebooks/Exploratory_analysis.py`: data quality checks and exploratory profiling
- `notebooks/Static_Funnel_Analysis.py`: static funnel metrics, segment analysis, duration analysis
- `notebooks/Temporal_funnel_analysis.py`: monthly trends for entries, conversion rates, durations

### Stage/step scopes available in static funnel notebook

When using `analyze_funnel(..., step=...)`, valid `step` values are:
- `created_to_mql`
- `mql_to_sql`
- `mql_to_won`
- `sql_to_opp`
- `sql_to_won`
- `opp_to_won`

Use `step=None` for full-funnel analysis.

## 5) Run the main scoring workflow

```bash
uv run python main.py
```

This executes the end-to-end scoring pipeline:
- read + clean data
- preprocess features
- train model
- print evaluation metrics

Default scoring stage in `main.py` is `FunnelStage.MQL`.
Supported model pipeline start stages are defined in `lead_scoring/registry.py`:
- `CREATED`
- `MQL`
- `SQL`
- `OPPORTUNITY`

## 6) Run tests

```bash
uv run pytest
```

## 7) Project structure

```text
Assignment-Jet-HR/
├── data/
│   └── lead_data.csv
├── lead_scoring/
│   ├── analysis/        # funnel metrics + plotting APIs (static and temporal)
│   ├── data/            # IO, cleaning, schema, validation, feature prep helpers
│   ├── scoring_model/   # config, preprocessing, training, evaluation
│   ├── tests/           # unit tests for cleaning/model pipeline
│   └── registry.py      # enums and model registry
├── notebooks/           # marimo notebooks as .py apps
├── main.py              # end-to-end lead scoring entrypoint
├── pyproject.toml       # dependencies and project metadata
└── uv.lock              # lockfile managed by uv```
