# statstools

Statistical tools for A/B experiment analysis — significance testing, confidence intervals, power analysis, and an interactive power analysis UI.

## Quick start

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Power Analysis UI

An interactive Streamlit app for experiment sizing and MDE calculation.

```bash
.venv/bin/streamlit run power_analysis_ui.py
```

Opens at **http://localhost:8501**

**Supported metric types:**

| Metric | Statistical test |
|--------|-----------------|
| Binary / Proportion (CTR, conversion) | Two-proportion z-test |
| Continuous (revenue, time, counts) | Welch t-test |
| Non-parametric | Mann-Whitney U (ARE ≈ 0.955 correction) |

**Two calculation modes:**
- **Required sample size** — enter your target MDE, get required n per variant + power curve.
  Also enter **total daily traffic** to get the estimated experiment run time in days and weeks.
- **MDE from traffic** — enter available users, get the minimum detectable effect + MDE-vs-sample-size chart.

Both modes support two-sided and **non-inferiority** (one-sided) test directions.

## Core library — `exp_tools/stat_tools.py`

### Main entry points

| Function | Purpose |
|----------|---------|
| `get_results(df, metric_field, ...)` | Full pipeline: p-value, summary stats, CI plot |
| `get_results_per_clienttype(...)` | Same, split by `clienttype_grouped` (web / app / web\|app) |
| `get_results_bootstrap(...)` | Non-parametric path: Mann-Whitney + bootstrap CI |
| `calculate_req_traffic_for_power(alpha, beta, base_rate, expected_effect)` | Required n per variant for a binary metric |
| `calculate_mde_from_traffic(alpha, beta, base_rate, n_variant)` | MDE given available traffic for a binary metric |

### Internal pipeline

`get_results` calls these in order:

1. **`get_df_stats`** — aggregates raw experiment data into a two-row summary (`visitors`, `reached_goal`, `stdv`, `binary` flag).
2. **`get_stats`** — selects test based on metric type: G-test (log-likelihood χ²) for binary, Welch t-test for continuous. Supports `non_inferiority_threshold` for one-sided tests.
3. **`get_ci`** — dispatches to the appropriate CI method:
   - `improv_interval_binomial` → `confidence_interval_ratio` (Fieller) for binary
   - `confidence_interval_ratio` (Fieller) for continuous ratio CIs
   - `confidence_interval_mean_differences` (Welch/Satterthwaite) for absolute CIs
4. **`plot_ci`** — Plotly horizontal CI chart (green = significant positive, red = significant negative, blue = not significant).

### Data contract

All functions expect a DataFrame with:
- `variant`: integer `0` (control) or `1` (treatment)
- A metric column: binary (`0`/`1`) or continuous

Binary vs. continuous is auto-detected via `np.isin(df[metric_field].unique(), [0, 1]).all()`.

## Tests

```bash
.venv/bin/python -m pytest tests/
```

## Notebooks

| Notebook | Description |
|----------|-------------|
| `experiment_simulations_power.ipynb` | Power calculations, simulations, CI distributions, A/A tests |
| `experiment_simulations_power_AA.ipynb` | A/A simulation and optional stopping / peeking analysis |
| `experiment_simulations_power_explained.ipynb` | Conceptual walkthrough of power and MDE |
