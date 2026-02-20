# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
.venv/bin/python -m pytest tests/

# Run a single test class or test
.venv/bin/python -m pytest tests/test_stat_tools.py::TestConfidenceIntervalRatio
.venv/bin/python -m pytest tests/test_stat_tools.py::TestGetStats::test_binary_metric_uses_g_test

# Run notebooks (JupyterLab or classic Jupyter)
jupyter lab
```

The virtual environment lives at `.venv/`.

## Architecture

### `exp_tools/stat_tools.py` — core library

All statistical logic lives here and is imported into notebooks via `from exp_tools.stat_tools import *`.

**Main public entry points:**

| Function | Purpose |
|----------|---------|
| `get_results(data, metric_field, ...)` | Full analysis pipeline: prints p-value, df_stats table, and renders a CI plot |
| `get_results_per_clienttype(...)` | Same as above, split by `clienttype_grouped` column (`web\|app`, `web`, `app`) |
| `get_results_bootstrap(...)` | Non-parametric path: Mann-Whitney + bootstrap CI |

**Internal pipeline** (`get_results` calls these in order):
1. `get_df_stats(df, metric_field)` — aggregates raw experiment DataFrame (must have columns `variant` ∈ {0,1} and the metric) into a two-row summary with `visitors`, `reached_goal`, `stdv`, and a `binary` flag (True when all metric values are 0/1).
2. `get_stats(df_stats, ...)` — selects the significance test based on `binary`: G-test (log-likelihood χ²) for binary metrics, Welch t-test for continuous. Supports a `non_inferioirity_threshold` for one-sided non-inferiority tests.
3. `get_ci(df_stats, metric_field, ...)` — dispatches to one of three CI methods:
   - `improv_interval_binomial` → `confidence_interval_ratio` (Fieller) for binary
   - `confidence_interval_ratio` (Fieller) for continuous ratio CIs
   - `confidence_interval_mean_differences` (Welch/Satterthwaite) for absolute CIs
4. `plot_ci(estimate, ci_l, ci_h, ...)` — renders a Plotly horizontal CI chart. Green = significant positive, red = significant negative, blue = not significant.

**Known issues in `stat_tools.py`:**
- `calculate_req_traffic_for_power` is defined **twice** — the second definition (lines 473–479, standard two-proportion z-test formula) silently overrides the first. The first definition uses a different MDE formula.
- `calculate_mde_from_traffic` references a global `BASE_RATE` instead of the `base_rate` parameter on the last line, making it context-dependent.

### Notebooks

The notebooks redefine `get_results` and `get_ci` locally as **silent versions** (no print statements, return dicts instead of printing) to enable use inside simulation loops. These local definitions shadow the ones imported from `stat_tools`.

- `experiment_simulations_power.ipynb` — original exploration notebook covering power calculations, power simulations, CI distributions, and A/A tests.
- `experiment_simulations_power_AA.ipynb` — focused notebook covering only the A/A simulation and optional stopping (peeking) analysis. This is the canonical reference for false positive rate and Type I error inflation.

### Data contract

All functions expect a DataFrame with:
- `variant`: integer 0 (control) or 1 (treatment)
- A metric column: binary (0/1) or continuous

`get_df_stats` auto-detects binary vs. continuous via `np.isin(df[metric_field].unique(), [0,1]).all()`.
