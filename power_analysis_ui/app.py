"""
Power Analysis UI — interactive Streamlit tool.

Supports:
  - Binary metrics    → two-proportion z-test
  - Continuous metrics → Welch t-test
  - Non-parametric    → Mann-Whitney (corrected via ARE)
  - Non-inferiority   → one-sided tests for all metric types

Two modes per metric type:
  1. Required sample size given MDE
  2. MDE given available traffic
"""

import sys
import os

# Ensure the repo root (parent of this folder) is on the path so that
# `exp_tools` can be imported regardless of where streamlit is invoked from.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from exp_tools.stat_tools import (
    calculate_mde_from_traffic,
    calculate_req_traffic_for_power,
)

# ── Constants ───────────────────────────────────────────────────────────────

METRIC_TYPES = {
    "Binary / Proportion": "binary",
    "Continuous (mean)": "continuous",
    "Non-parametric (Mann-Whitney)": "nonparametric",
}

TEST_DESCRIPTIONS = {
    "binary": "Two-proportion z-test (matches G-test for large n)",
    "continuous": "Welch t-test (two-sided or one-sided for non-inferiority)",
    "nonparametric": "Mann-Whitney U-test (uses ARE ≈ 0.955 efficiency correction vs t-test)",
}

# Asymptotic Relative Efficiency of Mann-Whitney vs t-test (normal distribution)
MANN_WHITNEY_ARE = 0.955

# ── Power calculation helpers ────────────────────────────────────────────────


def _z(p: float) -> float:
    return norm.ppf(p)


def sample_size_binary(
    alpha: float, beta: float, base_rate: float, mde_relative: float, one_sided: bool
) -> int:
    """Two-proportion z-test sample size per variant."""
    z_alpha = _z(1 - alpha / (1 if one_sided else 2))
    z_beta = _z(1 - beta)
    p1 = base_rate
    p2 = base_rate * (1 + mde_relative)
    p_bar = (p1 + p2) / 2
    n = (z_alpha * np.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (
        p2 - p1
    ) ** 2
    return int(np.ceil(n))


def mde_binary(alpha: float, beta: float, base_rate: float, n: int, one_sided: bool) -> float:
    """MDE (relative) for binary metric given n per variant."""
    z_alpha = _z(1 - alpha / (1 if one_sided else 2))
    z_beta = _z(1 - beta)
    sigma = np.sqrt(base_rate * (1 - base_rate))
    abs_mde = np.sqrt(2 * sigma**2 * (z_alpha + z_beta) ** 2 / n)
    return abs_mde / base_rate


def sample_size_continuous(
    alpha: float, beta: float, base_mean: float, base_std: float, mde_relative: float, one_sided: bool
) -> int:
    """Welch t-test sample size per variant."""
    z_alpha = _z(1 - alpha / (1 if one_sided else 2))
    z_beta = _z(1 - beta)
    delta = base_mean * mde_relative
    n = 2 * base_std**2 * (z_alpha + z_beta) ** 2 / delta**2
    return int(np.ceil(n))


def mde_continuous(alpha: float, beta: float, base_mean: float, base_std: float, n: int, one_sided: bool) -> float:
    """MDE (relative) for continuous metric given n per variant."""
    z_alpha = _z(1 - alpha / (1 if one_sided else 2))
    z_beta = _z(1 - beta)
    abs_mde = np.sqrt(2 * base_std**2 * (z_alpha + z_beta) ** 2 / n)
    return abs_mde / base_mean


def sample_size_nonparametric(
    alpha: float, beta: float, base_mean: float, base_std: float, mde_relative: float, one_sided: bool
) -> int:
    """Mann-Whitney sample size per variant (ARE-corrected t-test formula)."""
    n_ttest = sample_size_continuous(alpha, beta, base_mean, base_std, mde_relative, one_sided)
    return int(np.ceil(n_ttest / MANN_WHITNEY_ARE))


def mde_nonparametric(
    alpha: float, beta: float, base_mean: float, base_std: float, n: int, one_sided: bool
) -> float:
    """MDE (relative) for Mann-Whitney given n per variant (ARE-corrected)."""
    n_equivalent = int(n * MANN_WHITNEY_ARE)
    return mde_continuous(alpha, beta, base_mean, base_std, n_equivalent, one_sided)


# ── Power curve ─────────────────────────────────────────────────────────────


def power_curve_binary(alpha: float, base_rate: float, n: int, mde_relative: float, one_sided: bool):
    effects = np.linspace(0.001, max(mde_relative * 3, 0.3), 200)
    powers = []
    z_alpha = _z(1 - alpha / (1 if one_sided else 2))
    for eff in effects:
        p1, p2 = base_rate, base_rate * (1 + eff)
        se = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / n)
        delta = abs(p2 - p1)
        power = 1 - norm.cdf(z_alpha - delta / se)
        if not one_sided:
            power += norm.cdf(-z_alpha - delta / se)
        powers.append(min(power, 1.0))
    return effects * 100, powers


def power_curve_continuous(alpha: float, base_std: float, n: int, mde_relative: float, base_mean: float, one_sided: bool):
    effects = np.linspace(0.001, max(mde_relative * 3, 0.3), 200)
    powers = []
    z_alpha = _z(1 - alpha / (1 if one_sided else 2))
    for eff in effects:
        delta = base_mean * eff
        se = np.sqrt(2 * base_std**2 / n)
        power = 1 - norm.cdf(z_alpha - delta / se)
        if not one_sided:
            power += norm.cdf(-z_alpha - delta / se)
        powers.append(min(power, 1.0))
    return effects * 100, powers


def build_power_plot(effects_pct, powers, target_mde_pct, target_power, metric_label):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=effects_pct,
            y=[p * 100 for p in powers],
            mode="lines",
            name="Power curve",
            line=dict(color="#4C78A8", width=2.5),
        )
    )
    fig.add_hline(
        y=target_power * 100,
        line_dash="dash",
        line_color="green",
        annotation_text=f"Target power {target_power:.0%}",
        annotation_position="bottom right",
    )
    fig.add_vline(
        x=target_mde_pct,
        line_dash="dot",
        line_color="orange",
        annotation_text=f"MDE {target_mde_pct:.1f}%",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Power Curve — {metric_label}",
        xaxis_title="Relative Effect (%)",
        yaxis_title="Statistical Power (%)",
        yaxis=dict(range=[0, 105]),
        height=380,
        margin=dict(t=50, b=40, l=50, r=20),
        template="plotly_white",
    )
    return fig


# ── Sample-size sweep plot ───────────────────────────────────────────────────


def build_sample_sweep_plot(alpha, beta, base_rate, base_mean, base_std, mde_relative, one_sided, metric_type):
    ns = np.linspace(100, max(5_000_000, 20 * sample_size_binary(alpha, beta, base_rate, mde_relative, one_sided) if metric_type == "binary" else 20 * sample_size_continuous(alpha, beta, base_mean, base_std, mde_relative, one_sided)), 300).astype(int)
    mdes = []
    for n in ns:
        if metric_type == "binary":
            mdes.append(mde_binary(alpha, beta, base_rate, n, one_sided) * 100)
        elif metric_type == "continuous":
            mdes.append(mde_continuous(alpha, beta, base_mean, base_std, n, one_sided) * 100)
        else:
            mdes.append(mde_nonparametric(alpha, beta, base_mean, base_std, n, one_sided) * 100)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ns,
            y=mdes,
            mode="lines",
            name="MDE",
            line=dict(color="#E45756", width=2.5),
        )
    )
    fig.update_layout(
        title="MDE vs Sample Size (per variant)",
        xaxis_title="Sample Size per Variant",
        yaxis_title="MDE (%)",
        height=330,
        margin=dict(t=50, b=40, l=50, r=20),
        template="plotly_white",
    )
    return fig


# ── Streamlit App ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Power Analysis", page_icon="⚡", layout="wide")

st.title("⚡ A/B Test Power Analysis")
st.caption("Supports binary, continuous, and non-parametric metrics · two-sided and non-inferiority tests")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("🔧 Configuration")

    metric_label = st.selectbox("Metric type", list(METRIC_TYPES.keys()))
    metric_type = METRIC_TYPES[metric_label]

    st.markdown(f"**Test:** {TEST_DESCRIPTIONS[metric_type]}")
    st.divider()

    test_direction = st.radio("Test direction", ["Two-sided", "Non-inferiority (one-sided)"])
    one_sided = test_direction == "Non-inferiority (one-sided)"

    st.divider()
    st.subheader("Statistical thresholds")
    alpha = st.slider("Significance level (α)", 0.01, 0.20, 0.05, 0.01,
                      help="Type I error rate — probability of a false positive")
    power_target = st.slider("Target power (1−β)", 0.50, 0.99, 0.80, 0.05,
                             help="Probability of detecting a true effect")
    beta = 1 - power_target

    st.divider()
    st.subheader("Metric baseline")

    if metric_type == "binary":
        base_rate = st.number_input("Baseline conversion rate", 0.001, 0.999, 0.10, 0.005, format="%.3f")
        base_mean = base_rate
        base_std = np.sqrt(base_rate * (1 - base_rate))
    else:
        base_mean = st.number_input("Baseline mean", 0.01, 1e9, 50.0, help="e.g. average order value")
        base_std = st.number_input("Baseline std dev", 0.01, 1e9, 25.0, help="Standard deviation of the metric")
        base_rate = 0.1  # unused for non-binary but keeps function signatures clean

    st.divider()
    calc_mode = st.radio("Calculate", ["Required sample size", "MDE from traffic"])

# ── Main panel ───────────────────────────────────────────────────────────────

col_inputs, col_results = st.columns([1, 1], gap="large")

with col_inputs:
    st.subheader("Inputs")

    if calc_mode == "Required sample size":
        mde_pct = st.number_input(
            "Minimum Detectable Effect (%)",
            0.1, 200.0, 5.0, 0.5,
            help="Relative change you want to detect, e.g. 5 means +5% lift",
        )
        mde_relative = mde_pct / 100
        n_variants = st.number_input("Number of variants (incl. control)", 2, 10, 2)

        st.divider()
        st.markdown("**⏱ Runtime estimation**")
        daily_traffic = st.number_input(
            "Total daily traffic (all variants)",
            min_value=1,
            max_value=100_000_000,
            value=10_000,
            step=500,
            help="Total users entering the experiment per day across all variants",
        )

    else:
        n_available = st.number_input(
            "Available users per variant",
            100, 100_000_000, 10_000, 500,
        )
        mde_pct = None
        n_variants = st.number_input("Number of variants (incl. control)", 2, 10, 2)

    if one_sided:
        st.info(
            "🔁 **Non-inferiority mode**: one-sided test. "
            "The MDE/effect represents the minimum degradation you want to be able to detect."
        )

with col_results:
    st.subheader("Results")

    if calc_mode == "Required sample size":
        if metric_type == "binary":
            n_per_variant = sample_size_binary(alpha, beta, base_rate, mde_relative, one_sided)
        elif metric_type == "continuous":
            n_per_variant = sample_size_continuous(alpha, beta, base_mean, base_std, mde_relative, one_sided)
        else:
            n_per_variant = sample_size_nonparametric(alpha, beta, base_mean, base_std, mde_relative, one_sided)

        total_n = n_per_variant * n_variants
        days_required = int(np.ceil(total_n / daily_traffic))
        weeks_required = days_required / 7

        r1, r2, r3 = st.columns(3)
        r1.metric("Per variant", f"{n_per_variant:,}")
        r2.metric("Total (all variants)", f"{total_n:,}")
        r3.metric("MDE", f"{mde_pct:.1f}%")

        rt1, rt2, rt3 = st.columns(3)
        rt1.metric("⏱ Run time", f"{days_required} days")
        rt2.metric("≈ Weeks", f"{weeks_required:.1f}")
        rt3.metric("Daily traffic used", f"{daily_traffic:,}")

        # Power curve
        if metric_type == "binary":
            eff_x, pw_y = power_curve_binary(alpha, base_rate, n_per_variant, mde_relative, one_sided)
        else:
            eff_x, pw_y = power_curve_continuous(alpha, base_std, n_per_variant, mde_relative, base_mean, one_sided)
            if metric_type == "nonparametric":
                pw_y = [min(p / MANN_WHITNEY_ARE, 1.0) for p in pw_y]

        st.plotly_chart(
            build_power_plot(eff_x, pw_y, mde_pct, power_target, metric_label),
            use_container_width=True,
        )

    else:
        n_available = int(n_available)
        if metric_type == "binary":
            detected_mde = mde_binary(alpha, beta, base_rate, n_available, one_sided)
        elif metric_type == "continuous":
            detected_mde = mde_continuous(alpha, beta, base_mean, base_std, n_available, one_sided)
        else:
            detected_mde = mde_nonparametric(alpha, beta, base_mean, base_std, n_available, one_sided)

        detected_mde_pct = detected_mde * 100

        r1, r2 = st.columns(2)
        r1.metric("MDE (relative)", f"{detected_mde_pct:.2f}%")
        if metric_type == "binary":
            abs_mde = base_rate * detected_mde
            r2.metric("MDE (absolute)", f"{abs_mde:.4f}")
        else:
            abs_mde = base_mean * detected_mde
            r2.metric("MDE (absolute)", f"{abs_mde:.4f}")

        # MDE vs sample size sweep
        st.plotly_chart(
            build_sample_sweep_plot(
                alpha, beta, base_rate, base_mean, base_std, detected_mde, one_sided, metric_type
            ),
            use_container_width=True,
        )

# ── Summary table ─────────────────────────────────────────────────────────────

st.divider()
st.subheader("📋 Assumption Summary")

summary_rows = []

if calc_mode == "Required sample size":
    for pw in [0.70, 0.80, 0.90, 0.95]:
        b = 1 - pw
        if metric_type == "binary":
            n = sample_size_binary(alpha, b, base_rate, mde_relative, one_sided)
        elif metric_type == "continuous":
            n = sample_size_continuous(alpha, b, base_mean, base_std, mde_relative, one_sided)
        else:
            n = sample_size_nonparametric(alpha, b, base_mean, base_std, mde_relative, one_sided)
        total = n * int(n_variants)
        days = int(np.ceil(total / daily_traffic))
        summary_rows.append({
            "Power": f"{pw:.0%}",
            "α": f"{alpha:.2f}",
            "MDE": f"{mde_pct:.1f}%",
            "n per variant": f"{n:,}",
            "Total n": f"{total:,}",
            "Run time (days)": days,
            "Run time (weeks)": f"{days / 7:.1f}",
        })
else:
    for eff_pct in [1, 2, 5, 10, 15, 20]:
        eff = eff_pct / 100
        if metric_type == "binary":
            n = sample_size_binary(alpha, beta, base_rate, eff, one_sided)
        elif metric_type == "continuous":
            n = sample_size_continuous(alpha, beta, base_mean, base_std, eff, one_sided)
        else:
            n = sample_size_nonparametric(alpha, beta, base_mean, base_std, eff, one_sided)
        summary_rows.append({
            "MDE": f"{eff_pct}%",
            "α": f"{alpha:.2f}",
            "Power": f"{power_target:.0%}",
            "n per variant": f"{n:,}",
            "Total n": f"{n * int(n_variants):,}",
        })

import pandas as pd
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.caption(
    "Statistical tests aligned with `stat_tools.py`: "
    "G-test for binary · Welch t-test for continuous · Mann-Whitney (ARE≈0.955) for non-parametric"
)
