import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_tools.stat_tools import (
    one_sided_t_test,
    get_g_test,
    confidence_interval_mean_differences,
    get_df_stats,
    improv_interval_binomial,
    calculate_req_traffic_for_power,
    confidence_interval_ratio,
    get_mann_whitney_test,
    get_stats,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df_stats(
    visitors_base=1000,
    visitors_var=1000,
    reached_base=100,
    reached_var=120,
    stdv_base=0.3,
    stdv_var=0.3,
    binary=False,
):
    """Build a df_stats DataFrame matching the shape produced by get_df_stats."""
    return pd.DataFrame({
        "variant":       [0, 1],
        "visitors":      [visitors_base, visitors_var],
        "reached_goal":  [reached_base, reached_var],
        "average_value": [reached_base / visitors_base, reached_var / visitors_var],
        "stdv":          [stdv_base, stdv_var],
        "binary":        [binary, binary],
    })


def make_raw_df(n=500, diff=0.0, seed=42):
    """Build a raw experiment DataFrame with two variants."""
    rng = np.random.default_rng(seed)
    base   = rng.normal(loc=10.0,        scale=2.0, size=n)
    var    = rng.normal(loc=10.0 + diff, scale=2.0, size=n)
    return pd.DataFrame({
        "variant": [0] * n + [1] * n,
        "metric":  list(base) + list(var),
    })


# ---------------------------------------------------------------------------
# get_df_stats
# ---------------------------------------------------------------------------

class TestGetDfStats:
    def test_columns(self):
        df = make_raw_df()
        result = get_df_stats(df, "metric")
        assert list(result.columns) == ["variant", "visitors", "reached_goal", "average_value", "stdv", "binary"]

    def test_visitor_counts(self):
        df = make_raw_df(n=300)
        result = get_df_stats(df, "metric")
        assert result.at[0, "visitors"] == 300
        assert result.at[1, "visitors"] == 300

    def test_binary_flag_true(self):
        df = pd.DataFrame({
            "variant": [0, 0, 1, 1],
            "metric":  [0, 1, 0, 1],
        })
        result = get_df_stats(df, "metric")
        assert result["binary"].all() == True

    def test_binary_flag_false(self):
        df = make_raw_df()
        result = get_df_stats(df, "metric")
        assert result["binary"].all() == False

    def test_reached_goal_equals_sum(self):
        df = pd.DataFrame({
            "variant": [0, 0, 0, 1, 1, 1],
            "metric":  [1, 0, 1, 0, 1, 1],
        })
        result = get_df_stats(df, "metric")
        assert result.at[0, "reached_goal"] == 2
        assert result.at[1, "reached_goal"] == 2

    def test_two_variants_indexed(self):
        df = make_raw_df()
        result = get_df_stats(df, "metric")
        assert set(result["variant"]) == {0, 1}


# ---------------------------------------------------------------------------
# confidence_interval_ratio
# ---------------------------------------------------------------------------

class TestConfidenceIntervalRatio:
    def test_zero_base_returns_none(self):
        estimate, ci_l, ci_h = confidence_interval_ratio(
            avg_base=0, stdev_base=0.1, obs_base=1000,
            avg_var=0.1, stdev_var=0.1, obs_var=1000,
        )
        assert estimate is None
        assert ci_l is None
        assert ci_h is None

    def test_none_avg_var_returns_none(self):
        estimate, ci_l, ci_h = confidence_interval_ratio(
            avg_base=0.1, stdev_base=0.1, obs_base=1000,
            avg_var=None, stdev_var=0.1, obs_var=1000,
        )
        assert estimate is None

    def test_estimate_formula(self):
        avg_base, avg_var = 0.10, 0.12
        estimate, _, _ = confidence_interval_ratio(
            avg_base=avg_base, stdev_base=0.1, obs_base=1000,
            avg_var=avg_var,  stdev_var=0.1, obs_var=1000,
        )
        expected = (avg_var - avg_base) / abs(avg_base)
        assert abs(estimate - expected) < 1e-9

    def test_ci_ordered(self):
        estimate, ci_l, ci_h = confidence_interval_ratio(
            avg_base=0.10, stdev_base=0.1, obs_base=1000,
            avg_var=0.12,  stdev_var=0.1, obs_var=1000,
            confidence=0.95,
        )
        assert ci_l < ci_h

    def test_ci_contains_estimate(self):
        estimate, ci_l, ci_h = confidence_interval_ratio(
            avg_base=0.10, stdev_base=0.1, obs_base=1000,
            avg_var=0.12,  stdev_var=0.1, obs_var=1000,
            confidence=0.95,
        )
        assert ci_l <= estimate <= ci_h

    def test_wider_ci_at_higher_confidence(self):
        kwargs = dict(avg_base=0.10, stdev_base=0.1, obs_base=1000,
                      avg_var=0.12,  stdev_var=0.1, obs_var=1000)
        _, lo_90, hi_90 = confidence_interval_ratio(**kwargs, confidence=0.90)
        _, lo_99, hi_99 = confidence_interval_ratio(**kwargs, confidence=0.99)
        assert (hi_99 - lo_99) > (hi_90 - lo_90)

    def test_negative_base_average(self):
        # When avg_base < 0 the function negates fieller bounds (ci_l may exceed ci_h)
        estimate, ci_l, ci_h = confidence_interval_ratio(
            avg_base=-0.10, stdev_base=0.1, obs_base=1000,
            avg_var=-0.08,  stdev_var=0.1, obs_var=1000,
            confidence=0.90,
        )
        # Both bounds should be returned as finite floats
        assert ci_l is not None and ci_h is not None
        assert np.isfinite(ci_l) and np.isfinite(ci_h)


# ---------------------------------------------------------------------------
# confidence_interval_mean_differences
# ---------------------------------------------------------------------------

class TestConfidenceIntervalMeanDifferences:
    def test_delta_is_var_minus_base(self):
        avg_base, avg_var = 10.0, 12.0
        delta, _, _ = confidence_interval_mean_differences(
            confidence=0.95,
            avg_base=avg_base, avg_var=avg_var,
            stdev_base=2.0, stdev_var=2.0,
            obs_base=1000, obs_var=1000,
        )
        assert abs(delta - (avg_var - avg_base)) < 1e-9

    def test_ci_ordered(self):
        _, ci_l, ci_h = confidence_interval_mean_differences(
            confidence=0.95,
            avg_base=10.0, avg_var=12.0,
            stdev_base=2.0, stdev_var=2.0,
            obs_base=1000, obs_var=1000,
        )
        assert ci_l < ci_h

    def test_ci_contains_delta(self):
        delta, ci_l, ci_h = confidence_interval_mean_differences(
            confidence=0.95,
            avg_base=10.0, avg_var=12.0,
            stdev_base=2.0, stdev_var=2.0,
            obs_base=1000, obs_var=1000,
        )
        assert ci_l <= delta <= ci_h

    def test_equal_means_delta_zero(self):
        delta, ci_l, ci_h = confidence_interval_mean_differences(
            confidence=0.95,
            avg_base=10.0, avg_var=10.0,
            stdev_base=2.0, stdev_var=2.0,
            obs_base=1000, obs_var=1000,
        )
        assert abs(delta) < 1e-9
        assert ci_l < 0 < ci_h  # CI straddles zero

    def test_large_sample_narrow_ci(self):
        _, lo_small, hi_small = confidence_interval_mean_differences(
            confidence=0.95,
            avg_base=10.0, avg_var=12.0,
            stdev_base=2.0, stdev_var=2.0,
            obs_base=100, obs_var=100,
        )
        _, lo_large, hi_large = confidence_interval_mean_differences(
            confidence=0.95,
            avg_base=10.0, avg_var=12.0,
            stdev_base=2.0, stdev_var=2.0,
            obs_base=10000, obs_var=10000,
        )
        assert (hi_large - lo_large) < (hi_small - lo_small)


# ---------------------------------------------------------------------------
# improv_interval_binomial
# ---------------------------------------------------------------------------

class TestImprovIntervalBinomial:
    def test_zero_obs_base_returns_nan(self):
        result = improv_interval_binomial(
            confidence=0.95, successes_base=0, successes_var=50,
            obs_base=0, obs_var=100,
        )
        assert all(np.isnan(v) for v in result)

    def test_zero_obs_var_returns_nan(self):
        result = improv_interval_binomial(
            confidence=0.95, successes_base=50, successes_var=0,
            obs_base=100, obs_var=0,
        )
        assert all(np.isnan(v) for v in result)

    def test_successes_exceed_obs_returns_nan(self):
        result = improv_interval_binomial(
            confidence=0.95, successes_base=200, successes_var=50,
            obs_base=100, obs_var=100,
        )
        assert all(np.isnan(v) for v in result)

    def test_valid_input_returns_tuple(self):
        result = improv_interval_binomial(
            confidence=0.95, successes_base=50, successes_var=60,
            obs_base=1000, obs_var=1000,
        )
        assert len(result) == 3

    def test_positive_effect_positive_estimate(self):
        estimate, _, _ = improv_interval_binomial(
            confidence=0.90, successes_base=100, successes_var=120,
            obs_base=1000, obs_var=1000,
        )
        assert estimate > 0

    def test_ci_ordered(self):
        _, ci_l, ci_h = improv_interval_binomial(
            confidence=0.90, successes_base=100, successes_var=120,
            obs_base=1000, obs_var=1000,
        )
        assert ci_l < ci_h


# ---------------------------------------------------------------------------
# get_g_test
# ---------------------------------------------------------------------------

class TestGetGTest:
    def test_returns_value_between_0_and_1(self):
        p = get_g_test(
            counts_base=100, counts_var=110,
            visitors_base=1000, visitors_var=1000,
        )
        assert 0 <= p <= 1

    def test_equal_rates_high_pvalue(self):
        # Same conversion rate → should NOT be significant
        p = get_g_test(
            counts_base=100, counts_var=100,
            visitors_base=1000, visitors_var=1000,
        )
        assert p > 0.5

    def test_very_different_rates_low_pvalue(self):
        # Wildly different rates → should be very significant
        p = get_g_test(
            counts_base=10,  counts_var=990,
            visitors_base=1000, visitors_var=1000,
        )
        assert p < 0.001


# ---------------------------------------------------------------------------
# one_sided_t_test
# ---------------------------------------------------------------------------

class TestOneSidedTTest:
    def test_returns_pvalue_in_range(self):
        p = one_sided_t_test(
            estimate_base=0.10, sd_base=0.30, n_base=1000,
            estimate_variant=0.12, sd_variant=0.30, n_variant=1000,
            acceptable_cost_value=0.0,
        )
        assert 0 <= p <= 1

    def test_large_improvement_small_pvalue(self):
        # Variant is much better; one-sided p should be small
        p = one_sided_t_test(
            estimate_base=0.10, sd_base=0.05, n_base=100_000,
            estimate_variant=0.20, sd_variant=0.05, n_variant=100_000,
            acceptable_cost_value=0.0,
            acceptable_cost_type="relative",
            increase_is="good",
        )
        assert p < 0.01

    def test_no_improvement_high_pvalue(self):
        p = one_sided_t_test(
            estimate_base=0.10, sd_base=0.30, n_base=1000,
            estimate_variant=0.10, sd_variant=0.30, n_variant=1000,
            acceptable_cost_value=0.0,
        )
        assert p > 0.4

    def test_relative_vs_impact_cost_type(self):
        common = dict(
            estimate_base=0.10, sd_base=0.30, n_base=1000,
            estimate_variant=0.12, sd_variant=0.30, n_variant=1000,
        )
        p_rel    = one_sided_t_test(**common, acceptable_cost_value=0.05, acceptable_cost_type="relative")
        p_impact = one_sided_t_test(**common, acceptable_cost_value=10.0,  acceptable_cost_type="impact")
        # Both should be valid probabilities
        assert 0 <= p_rel    <= 1
        assert 0 <= p_impact <= 1

    def test_increase_is_bad_flips_direction(self):
        # Use a non-zero cost so thresholds diverge between directions
        common = dict(
            estimate_base=0.10, sd_base=0.30, n_base=1000,
            estimate_variant=0.12, sd_variant=0.30, n_variant=1000,
            acceptable_cost_value=0.05,
            acceptable_cost_type="relative",
        )
        p_good = one_sided_t_test(**common, increase_is="good")
        p_bad  = one_sided_t_test(**common, increase_is="bad")
        # With variant > base, "good" (lower threshold) gives a smaller p than "bad" (higher threshold)
        assert p_good < p_bad


# ---------------------------------------------------------------------------
# calculate_req_traffic_for_power
# ---------------------------------------------------------------------------

class TestCalculateReqTrafficForPower:
    def test_returns_positive_integer(self):
        n = calculate_req_traffic_for_power(alpha=0.05, beta=0.20, base_rate=0.10, expected_effect=0.05)
        assert isinstance(n, int)
        assert n > 0

    def test_larger_effect_needs_less_traffic(self):
        n_small = calculate_req_traffic_for_power(alpha=0.05, beta=0.20, base_rate=0.10, expected_effect=0.02)
        n_large = calculate_req_traffic_for_power(alpha=0.05, beta=0.20, base_rate=0.10, expected_effect=0.10)
        assert n_small > n_large

    def test_stricter_alpha_needs_more_traffic(self):
        n_loose  = calculate_req_traffic_for_power(alpha=0.10, beta=0.20, base_rate=0.10, expected_effect=0.05)
        n_strict = calculate_req_traffic_for_power(alpha=0.01, beta=0.20, base_rate=0.10, expected_effect=0.05)
        assert n_strict > n_loose

    def test_higher_power_needs_more_traffic(self):
        n_low  = calculate_req_traffic_for_power(alpha=0.05, beta=0.30, base_rate=0.10, expected_effect=0.05)
        n_high = calculate_req_traffic_for_power(alpha=0.05, beta=0.05, base_rate=0.10, expected_effect=0.05)
        assert n_high > n_low


# ---------------------------------------------------------------------------
# get_mann_whitney_test
# ---------------------------------------------------------------------------

class TestGetMannWhitneyTest:
    def test_returns_u_and_pvalue(self):
        df = make_raw_df(n=200)
        u, p = get_mann_whitney_test(df, "metric")
        assert u > 0
        assert 0 <= p <= 1

    def test_identical_distributions_high_pvalue(self):
        df = make_raw_df(n=500, diff=0.0)
        _, p = get_mann_whitney_test(df, "metric")
        assert p > 0.05

    def test_very_different_distributions_low_pvalue(self):
        df = make_raw_df(n=500, diff=5.0)
        _, p = get_mann_whitney_test(df, "metric")
        assert p < 0.01


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_returns_dict_with_expected_keys(self):
        df_stats = make_df_stats()
        result = get_stats(df_stats)
        assert "p-value" in result
        assert "N" in result
        assert "ratio var/base" in result

    def test_pvalue_in_range(self):
        df_stats = make_df_stats()
        result = get_stats(df_stats)
        assert 0 <= result["p-value"] <= 1

    def test_total_n(self):
        df_stats = make_df_stats(visitors_base=800, visitors_var=1200)
        result = get_stats(df_stats)
        assert result["N"] == 2000

    def test_binary_metric_uses_g_test(self):
        # Binary data path; just verify it runs and returns sensible values
        df_stats = make_df_stats(
            visitors_base=1000, visitors_var=1000,
            reached_base=100,   reached_var=120,
            binary=True,
        )
        result = get_stats(df_stats)
        assert 0 <= result["p-value"] <= 1

    def test_non_inferiority_threshold(self):
        df_stats = make_df_stats()
        result = get_stats(df_stats, non_inferioirity_threshold=-0.05)
        assert 0 <= result["p-value"] <= 1

    def test_ratio_var_base(self):
        df_stats = make_df_stats(
            visitors_base=1000, visitors_var=1000,
            reached_base=100,   reached_var=110,
        )
        result = get_stats(df_stats)
        expected_ratio = (110 / 1000) / (100 / 1000) - 1
        assert abs(result["ratio var/base"] - expected_ratio) < 1e-9
