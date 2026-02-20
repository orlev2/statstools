"""Statistical tools for A/B experiment analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional
from scipy.stats import chi2_contingency, chi2, ttest_ind_from_stats, norm, binom, mannwhitneyu, t, chisquare
from statsmodels.stats.proportion import proportion_confint
import statsmodels.formula.api as smf
import patsy
import plotly.graph_objects as go


def vertical_plot_lines(x: pd.Series, xal: float, yal: float, **kwargs) -> None:
    """Draw a vertical dashed line at the mean of *x* with a text annotation.

    Parameters
    ----------
    x : pd.Series
        Data whose mean determines the line position.
    xal : float
        Horizontal offset (in data units) applied to the annotation text.
    yal : float
        Vertical position (in data units) for the annotation text.
    **kwargs
        color : str, optional
            Line and text colour. Defaults to ``"g"``.
    """
    color = kwargs.get("color", "g")
    plt.axvline(x.mean(), linestyle="--", color=color, alpha=0.5)
    tx_mean = "mean: {:.3f}".format(x.mean())
    txkw = dict(size=11, color=color, rotation=90)
    plt.text(x.mean() + xal, yal, tx_mean, **txkw)


def plot_distribution(df: pd.DataFrame, metric: str, **kwargs) -> None:
    """Plot KDE distributions of a metric split by variant.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment DataFrame containing a ``variant`` column.
    metric : str
        Column name of the metric to plot.
    **kwargs
        row : optional
            FacetGrid row variable.
        displot_hist : bool
            Whether to overlay a histogram. Defaults to ``False``.
        kde_bandwidth : str or float
            KDE bandwidth argument. Defaults to ``"silverman"``.
        xal : float
            Horizontal offset for the mean annotation. Defaults to ``0.10``.
        yal : float
            Vertical position for the mean annotation. Defaults to ``0.5``.
        xlim : tuple[float, float]
            x-axis limits. Defaults to ``(-10, 10)``.
    """
    g = sns.FacetGrid(
        data=df,
        aspect=3,
        height=5,
        hue="variant",
        row=kwargs.get("row", None),
    )
    g.map(
        sns.distplot,
        metric,
        hist=kwargs.get("displot_hist", False),
        kde_kws={"bw": kwargs.get("kde_bandwidth", "silverman")},
    )
    g.map(
        vertical_plot_lines,
        metric,
        xal=kwargs.get("xal", 0.10),
        yal=kwargs.get("yal", 0.5),
    )
    g.fig.suptitle("Distribution between variants - {}".format(metric))
    g.set_xlabels("{}".format(metric))
    g.set_ylabels("density")
    g.set(xlim=kwargs.get("xlim", (-10, 10)))
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    for ax in g.axes:
        ax[0].axvline(x=0, color="black", ls=":")


def one_sided_t_test(
    estimate_base: float,
    sd_base: float,
    n_base: int,
    estimate_variant: float,
    sd_variant: float,
    n_variant: int,
    acceptable_cost_value: float,
    acceptable_cost_type: str = "relative",
    increase_is: str = "good",
) -> float:
    """Run a one-sided Welch t-test for non-inferiority or superiority.

    The null-hypothesis mean is shifted by ``acceptable_cost_value`` to test
    whether the variant is at most ``acceptable_cost_value`` worse than control
    (non-inferiority) or at least ``acceptable_cost_value`` better (superiority).

    Parameters
    ----------
    estimate_base : float
        Mean of the control group.
    sd_base : float
        Standard deviation of the control group.
    n_base : int
        Number of observations in the control group.
    estimate_variant : float
        Mean of the treatment group.
    sd_variant : float
        Standard deviation of the treatment group.
    n_variant : int
        Number of observations in the treatment group.
    acceptable_cost_value : float
        Magnitude of the allowed degradation.

        - ``acceptable_cost_type="relative"``: fraction of ``estimate_base``
          (e.g. ``0.05`` allows 5 % relative degradation).
        - ``acceptable_cost_type="impact"``: absolute total impact split
          across all observations.
    acceptable_cost_type : {"relative", "impact"}
        Interpretation of ``acceptable_cost_value``. Defaults to ``"relative"``.
    increase_is : {"good", "bad"}
        Direction of a beneficial effect. ``"good"`` tests that the variant
        is not worse than control; ``"bad"`` tests that it is not better.
        Defaults to ``"good"``.

    Returns
    -------
    float
        One-sided p-value.
    """
    if acceptable_cost_type == "relative":
        delta = acceptable_cost_value * estimate_base
    elif acceptable_cost_type == "impact":
        delta = acceptable_cost_value / (n_variant + n_base)

    if increase_is == "good":
        threshold_value = estimate_base - delta
    elif increase_is == "bad":
        threshold_value = estimate_base + delta

    t_test = ttest_ind_from_stats(
        mean1=threshold_value,
        std1=sd_base,
        nobs1=n_base,
        mean2=estimate_variant,
        std2=sd_variant,
        nobs2=n_variant,
        equal_var=False,
    )
    p_value = t_test[1] / 2.0
    return p_value


def get_g_test(
    counts_base: float,
    counts_var: float,
    visitors_base: int,
    visitors_var: int,
) -> float:
    """Compute the G-test (log-likelihood chi-squared) p-value for a 2×2 table.

    Tests whether the conversion rates differ between control and treatment.

    Parameters
    ----------
    counts_base : float
        Number of conversions in the control group.
    counts_var : float
        Number of conversions in the treatment group.
    visitors_base : int
        Total visitors in the control group.
    visitors_var : int
        Total visitors in the treatment group.

    Returns
    -------
    float
        Two-sided p-value. Returns ``np.nan`` if the table is degenerate.
    """
    try:
        p_value = chi2_contingency(
            [[counts_base, counts_var], [visitors_base - counts_base, visitors_var - counts_var]],
            correction=False,
            lambda_="log-likelihood",
        )[1]
    except ValueError:
        p_value = np.nan
    return p_value


def confidence_interval_mean_differences(
    confidence: float,
    avg_base: float,
    avg_var: float,
    stdev_base: float,
    stdev_var: float,
    obs_base: int,
    obs_var: int,
) -> tuple:
    """Compute a Welch/Satterthwaite CI for the absolute mean difference.

    Uses the Satterthwaite approximation for degrees of freedom.

    Parameters
    ----------
    confidence : float
        Confidence level (e.g. ``0.9`` for 90 %).
    avg_base : float
        Mean of the control group.
    avg_var : float
        Mean of the treatment group.
    stdev_base : float
        Standard deviation of the control group.
    stdev_var : float
        Standard deviation of the treatment group.
    obs_base : int
        Number of observations in the control group.
    obs_var : int
        Number of observations in the treatment group.

    Returns
    -------
    tuple[float, float, float]
        ``(delta, ci_l, ci_h)`` where ``delta = avg_var - avg_base`` and
        ``[ci_l, ci_h]`` is the confidence interval for ``delta``.
    """
    pooled_se = np.sqrt(stdev_base**2 / obs_base + stdev_var**2 / obs_var)
    delta = avg_var - avg_base

    # Satterthwaite degrees of freedom
    df = (stdev_base**2 / obs_base + stdev_var**2 / obs_var) ** 2 / (
        (stdev_base**2) ** 2 / (obs_base**2 * (obs_base - 1))
        + (stdev_var**2) ** 2 / (obs_var**2 * (obs_var - 1))
    )

    t_crit = t.ppf(1 - (1 - confidence) / 2, df)
    ci_l = delta - t_crit * pooled_se
    ci_h = delta + t_crit * pooled_se

    return delta, ci_l, ci_h


def plot_ci(
    estimate: float,
    ci_l: float,
    ci_h: float,
    metric_field: str,
    non_inferiority_threshold: Optional[float] = None,
    ratio: bool = True,
) -> None:
    """Render a Plotly horizontal confidence-interval chart.

    Colour coding by statistical significance:

    - **Blue** – not significant.
    - **Green** – significantly positive.
    - **Red** – significantly negative.

    For non-inferiority tests the upper error bar extends to a large constant
    to reflect the one-sided nature of the test.

    Parameters
    ----------
    estimate : float
        Point estimate (centre of the CI marker).
    ci_l : float
        Lower bound of the confidence interval.
    ci_h : float
        Upper bound of the confidence interval.
    metric_field : str
        Metric name, used as the chart title.
    non_inferiority_threshold : float, optional
        When provided, a vertical line is drawn at this threshold and the
        significance decision is made relative to it rather than zero.
        Defaults to ``None`` (standard two-sided test around zero).
    ratio : bool
        If ``False``, adds an annotation stating that axis units are absolute
        values. Defaults to ``True``.
    """
    significance_threshold = non_inferiority_threshold if non_inferiority_threshold is not None else 0

    is_significant = not (ci_l < significance_threshold < ci_h)

    if not is_significant:
        stat_text = "Effect is not statistically significant"
        color_marker = "rgb(70,130,180)"
        color_ci = "rgb(161,172,216)"
    else:
        stat_text = "Effect is statistically significant"
        if estimate > significance_threshold:
            color_marker = "rgb(0,100,0)"
            color_ci = "rgb(60,179,113)"
        else:
            color_marker = "rgb(128,0,0)"
            color_ci = "rgb(250,128,114)"

    fig = go.Figure()
    if non_inferiority_threshold is None:
        fig.add_trace(
            go.Scatter(
                y=[0],
                x=[estimate],
                mode="markers",
                marker_symbol="line-ns",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[ci_h - estimate],
                    arrayminus=[estimate - ci_l],
                    color=color_ci,
                    thickness=100,
                    width=0,
                ),
                hovertext=[stat_text],
                marker=dict(size=70, line_width=2, line_color=color_marker),
            )
        )
    else:
        # One-sided: upper bound extends to infinity, represented as a large constant
        fig.add_trace(
            go.Scatter(
                y=[0],
                x=[estimate],
                mode="markers",
                marker_symbol="line-ns",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[999],
                    arrayminus=[estimate - ci_l],
                    color=color_ci,
                    thickness=100,
                    width=0,
                ),
                opacity=1.0,
                hovertext=[stat_text],
                marker=dict(size=70, line_width=2, line_color=color_marker),
            )
        )

    fig.add_shape(
        dict(
            type="line",
            x0=0, y0=-1, x1=0, y1=1,
            line=dict(color="black", width=2.5, dash="dot"),
        )
    )

    if non_inferiority_threshold is not None:
        fig.add_shape(
            dict(
                type="line",
                x0=non_inferiority_threshold, y0=-1,
                x1=non_inferiority_threshold, y1=1,
                line=dict(color="black", width=4),
            )
        )

    annotations = [
        dict(
            x=estimate, y=-0.65,
            text=stat_text,
            showarrow=False,
            xanchor="left",
            xshift=10,
            opacity=0.7,
            font=dict(size=15),
        )
    ]
    if not ratio:
        annotations.append(
            dict(
                x=estimate, y=-0.78,
                text="Axis units are in absolute values",
                showarrow=False,
                xanchor="left",
                xshift=10,
                opacity=0.7,
                font=dict(size=12),
            )
        )

    fig.update_layout(
        title=metric_field,
        autosize=False,
        width=1000,
        height=400,
        annotations=annotations,
        font=dict(family="Courier New, monospace", size=18, color="#7f7f7f"),
    )
    fig.update_yaxes(showticklabels=False)

    if non_inferiority_threshold is not None:
        x_range = [min(-0.3, non_inferiority_threshold), 0.3]
    else:
        x_range = [-0.3, 0.3]

    if ci_l < x_range[0] or ci_h > x_range[1]:
        x_range = [min(ci_l - 0.05, -0.05), max(ci_h + 0.05, 0.05)]

    fig.update_xaxes(dict(range=x_range))
    fig.show()


def get_ci(
    df_stats: pd.DataFrame,
    metric_field: str,
    confidence: float = 0.9,
    plot: bool = True,
    non_inferiority_threshold: Optional[float] = None,
    ratio: bool = True,
) -> None:
    """Compute, print, and optionally plot the confidence interval for the treatment effect.

    Dispatches to the appropriate CI method based on metric type:

    - Binary metrics → Fieller's method via :func:`improv_interval_binomial`.
    - Continuous metrics with ``ratio=True`` → Fieller's method via
      :func:`confidence_interval_ratio`.
    - Continuous metrics with ``ratio=False`` → Welch/Satterthwaite via
      :func:`confidence_interval_mean_differences`.

    Parameters
    ----------
    df_stats : pd.DataFrame
        Two-row summary DataFrame produced by :func:`get_df_stats`.
        Row 0 = control, row 1 = treatment.
    metric_field : str
        Metric name, used for the plot title.
    confidence : float
        Confidence level (e.g. ``0.9`` for 90 %). Defaults to ``0.9``.
    plot : bool
        Whether to render the CI chart. Defaults to ``True``.
    non_inferiority_threshold : float, optional
        Non-inferiority margin as a relative fraction (e.g. ``-0.05`` for
        5 % allowed degradation). When provided, prints a one-sided decision.
        Defaults to ``None``.
    ratio : bool
        For continuous metrics, whether to report a relative (ratio) CI or
        an absolute difference CI. Defaults to ``True``.
    """
    if df_stats["binary"].all():
        estimate, ci_l, ci_h = improv_interval_binomial(
            confidence=confidence,
            successes_base=df_stats.at[0, "reached_goal"],
            successes_var=df_stats.at[1, "reached_goal"],
            obs_base=df_stats.at[0, "visitors"],
            obs_var=df_stats.at[1, "visitors"],
        )
        print("Estimate: {:.3%},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))
    elif ratio:
        estimate, ci_l, ci_h = confidence_interval_ratio(
            confidence=confidence,
            avg_base=df_stats.at[0, "reached_goal"] / df_stats.at[0, "visitors"],
            avg_var=df_stats.at[1, "reached_goal"] / df_stats.at[1, "visitors"],
            stdev_base=df_stats.at[0, "stdv"],
            stdev_var=df_stats.at[1, "stdv"],
            obs_base=df_stats.at[0, "visitors"],
            obs_var=df_stats.at[1, "visitors"],
        )
        print("Estimate: {:.3%},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))
    else:
        estimate, ci_l, ci_h = confidence_interval_mean_differences(
            confidence=confidence,
            avg_base=df_stats.at[0, "reached_goal"] / df_stats.at[0, "visitors"],
            avg_var=df_stats.at[1, "reached_goal"] / df_stats.at[1, "visitors"],
            stdev_base=df_stats.at[0, "stdv"],
            stdev_var=df_stats.at[1, "stdv"],
            obs_base=df_stats.at[0, "visitors"],
            obs_var=df_stats.at[1, "visitors"],
        )
        print("Estimate: {:.3f},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))

    if ci_h < ci_l:
        print("*** ci_h and ci_l are reversed - check why ***")
        ci_l, ci_h = min(ci_l, ci_h), max(ci_l, ci_h)

    if non_inferiority_threshold is not None:
        fmt = "{:.0%}" if ratio else "{:.2f}"
        print(
            ("CI_low > " + fmt + "? {}").format(non_inferiority_threshold, ci_l > non_inferiority_threshold)
        )

    if plot:
        plot_ci(
            estimate, ci_l, ci_h, metric_field,
            non_inferiority_threshold=non_inferiority_threshold,
            ratio=ratio,
        )


def get_mann_whitney_test(
    data: pd.DataFrame,
    metric_field: str,
    confidence: float = 0.9,
) -> tuple:
    """Run the Mann-Whitney U test for a location shift between variants.

    Parameters
    ----------
    data : pd.DataFrame
        Experiment DataFrame with ``variant`` (0/1) and ``metric_field`` columns.
    metric_field : str
        Column name of the metric to test.
    confidence : float
        Confidence level used to determine the significance threshold
        (prints whether ``p_value < 1 - confidence``). Defaults to ``0.9``.

    Returns
    -------
    tuple[float, float]
        ``(u_value, p_value)``.
    """
    array_base = data.loc[data["variant"] == 0, metric_field]
    array_variant = data.loc[data["variant"] == 1, metric_field]
    u_value, p_value = mannwhitneyu(array_base, array_variant)

    print(
        "Mann-Whitney p-value: {:.5f} \nstatistical significance at {} level p_value < (1 - confidence): {}\n".format(
            p_value, confidence, p_value < (1 - confidence)
        )
    )
    return u_value, p_value


def get_ci_bootstrap(
    data: pd.DataFrame,
    metric_field: str,
    confidence: float = 0.9,
    n_replicates: int = 100,
    plot: bool = True,
    non_inferiority_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """Estimate a non-parametric CI for the mean difference via bootstrap resampling.

    Resamples the full dataset with replacement ``n_replicates`` times,
    computes per-variant means for each replicate, then derives the CI from
    the empirical distribution of mean differences.

    Parameters
    ----------
    data : pd.DataFrame
        Experiment DataFrame with ``variant`` (0/1) and ``metric_field`` columns.
    metric_field : str
        Column name of the metric.
    confidence : float
        Confidence level (e.g. ``0.9`` for 90 %). Defaults to ``0.9``.
    n_replicates : int
        Number of bootstrap resamples. Defaults to ``100``.
    plot : bool
        Whether to render the CI chart. Defaults to ``True``.
    non_inferiority_threshold : float, optional
        Non-inferiority margin. When provided, prints a one-sided decision.
        Defaults to ``None``.

    Returns
    -------
    pd.DataFrame
        Concatenated per-replicate variant means (rows indexed by variant).
    """
    print("\n=== Bootstrap sampling with {} replications to get non-parametric CI ===".format(n_replicates))

    bootstrap_rows = []
    for _ in range(n_replicates):
        df_bootstrap = (
            data[["variant", metric_field]]
            .sample(len(data), replace=True)
            .groupby("variant")
            .mean()
            .reset_index()
        )
        bootstrap_rows.append(df_bootstrap)

    bootstraps_df = pd.concat(bootstrap_rows, ignore_index=True)

    bs_mean_diff = (
        np.array(bootstraps_df[bootstraps_df.variant == 1][metric_field])
        - np.array(bootstraps_df[bootstraps_df.variant == 0][metric_field])
    )

    estimate = bs_mean_diff.mean()
    ci_l = np.percentile(bs_mean_diff, 100 * (1 - confidence) / 2)
    ci_h = np.percentile(bs_mean_diff, 100 * (1 - (1 - confidence) / 2))

    print("Estimate: {:.3f},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))
    if non_inferiority_threshold is not None:
        print("CI_low > {:.2f}? {}".format(non_inferiority_threshold, ci_l > non_inferiority_threshold))

    plot_ci(
        estimate, ci_l, ci_h, metric_field,
        non_inferiority_threshold=non_inferiority_threshold,
        ratio=False,
    )

    return bootstraps_df


def get_stats(
    df_stats: pd.DataFrame,
    non_inferiority_threshold: Optional[float] = None,
) -> dict:
    """Select and run the appropriate significance test for the experiment.

    For binary metrics the G-test (log-likelihood chi-squared) is used.
    For continuous metrics Welch's t-test is used — two-sided by default,
    one-sided for non-inferiority by shifting the null hypothesis mean.

    Parameters
    ----------
    df_stats : pd.DataFrame
        Two-row summary DataFrame produced by :func:`get_df_stats`.
        Row 0 = control, row 1 = treatment.
    non_inferiority_threshold : float, optional
        Relative non-inferiority margin (e.g. ``-0.05`` for 5 % allowed
        degradation). When provided a one-sided p-value is reported.
        Defaults to ``None`` (two-sided test).

    Returns
    -------
    dict
        Keys:

        - ``"p-value"`` : float
        - ``"N"`` : total number of observations
        - ``"ratio var/base"`` : relative difference ``(avg_var / avg_base) - 1``
    """
    two_sided = non_inferiority_threshold is None

    avg_base = df_stats["reached_goal"][0] / df_stats["visitors"][0]
    avg_var = df_stats["reached_goal"][1] / df_stats["visitors"][1]

    if df_stats["binary"].all():
        p_val = get_g_test(
            counts_base=df_stats.at[0, "reached_goal"],
            counts_var=df_stats.at[1, "reached_goal"],
            visitors_base=df_stats.at[0, "visitors"],
            visitors_var=df_stats.at[1, "visitors"],
        )
        print("\npval = {:.5f}, significant at 10%: {}\n".format(p_val, p_val < 0.1))
    elif two_sided:
        p_val = ttest_ind_from_stats(
            mean1=avg_base,
            std1=df_stats["stdv"][0],
            nobs1=df_stats["visitors"][0],
            mean2=avg_var,
            std2=df_stats["stdv"][1],
            nobs2=df_stats["visitors"][1],
            equal_var=False,
        )[1]
        print("\npval = {:.5f}, significant at 10%: {}\n".format(p_val, p_val < 0.1))
    else:
        # One-sided non-inferiority: shift null mean by the threshold
        p_val = ttest_ind_from_stats(
            mean1=avg_base * (1 + non_inferiority_threshold),
            std1=df_stats["stdv"][0],
            nobs1=df_stats["visitors"][0],
            mean2=avg_var,
            std2=df_stats["stdv"][1],
            nobs2=df_stats["visitors"][1],
            equal_var=False,
        )[1] / 2
        print(
            "\nNon-inferiority threshold: {:.0%}, pval = {:.5f}, significant at 10%: {}\n".format(
                non_inferiority_threshold, p_val, p_val < 0.1
            )
        )

    return {
        "p-value": p_val,
        "N": df_stats.at[0, "visitors"] + df_stats.at[1, "visitors"],
        "ratio var/base": (avg_var / avg_base) - 1,
    }


def get_df_stats(df: pd.DataFrame, metric_field: str) -> pd.DataFrame:
    """Aggregate a raw experiment DataFrame into per-variant summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment DataFrame with a ``variant`` column (0 = control,
        1 = treatment) and a numeric ``metric_field`` column.
    metric_field : str
        Column name of the metric to aggregate.

    Returns
    -------
    pd.DataFrame
        Two-row DataFrame (row 0 = control, row 1 = treatment) with columns:

        - ``variant`` : 0 or 1
        - ``visitors`` : number of observations
        - ``reached_goal`` : sum of the metric (events / conversions)
        - ``average_value`` : mean of the metric
        - ``stdv`` : standard deviation of the metric
        - ``binary`` : ``True`` if the metric only takes values in {0, 1}
    """
    df_stats = (
        df.groupby("variant")
        .agg({metric_field: ["count", "sum", "mean", "std"]})
        .reset_index()
    )
    df_stats.columns = ["variant", "visitors", "reached_goal", "average_value", "stdv"]
    df_stats["binary"] = np.isin(df[metric_field].unique(), [0, 1]).all()
    return df_stats


def get_results_per_clienttype(
    data: pd.DataFrame,
    metric_field: str,
    confidence: float = 0.9,
    threshold: Optional[float] = None,
    calculate_ratio: bool = True,
) -> None:
    """Run the full analysis pipeline separately for each client-type group.

    Iterates over ``["web|app", "web", "app"]`` subsets of
    ``data["clienttype_grouped"]``, printing the p-value and CI for each.

    Parameters
    ----------
    data : pd.DataFrame
        Experiment DataFrame with ``variant``, ``clienttype_grouped``, and
        ``metric_field`` columns.
    metric_field : str
        Metric to analyse.
    confidence : float
        Two-sided confidence level. Defaults to ``0.9``.
    threshold : float, optional
        Non-inferiority margin. When provided, the one-sided confidence level
        used for the CI is ``1 - (1 - confidence) * 2``. Defaults to ``None``.
    calculate_ratio : bool
        Whether to compute a relative (ratio) CI. Defaults to ``True``.
    """
    # Compute the effective confidence level once so it stays consistent
    # across all client-type iterations.
    if threshold is not None:
        effective_confidence = 1 - (1 - confidence) * 2
        print(
            "One sided test: confidence level tested is: 1 - (1 - confidence)*2 = {}\n".format(
                effective_confidence
            )
        )
    else:
        effective_confidence = confidence

    print("{}\n\n======".format(metric_field))
    for ctg in ["web|app", "web", "app"]:
        print("\n" + ctg + "\n------")
        data_ctg = data[data["clienttype_grouped"].str.contains(ctg)]

        df_stats = get_df_stats(data_ctg, metric_field)
        get_stats(df_stats, non_inferiority_threshold=threshold)
        print("\n", df_stats, "\n")

        get_ci(
            df_stats,
            metric_field,
            confidence=effective_confidence,
            plot=True,
            non_inferiority_threshold=threshold,
            ratio=calculate_ratio,
        )


def get_results_bootstrap(
    data: pd.DataFrame,
    metric_field: str,
    confidence: float = 0.9,
    threshold: Optional[float] = None,
    **kwargs,
) -> None:
    """Non-parametric analysis pipeline: Mann-Whitney test + bootstrap CI.

    Parameters
    ----------
    data : pd.DataFrame
        Experiment DataFrame with ``variant`` (0/1) and ``metric_field`` columns.
    metric_field : str
        Metric to analyse.
    confidence : float
        Two-sided confidence level. Defaults to ``0.9``.
    threshold : float, optional
        Non-inferiority margin. When provided, the one-sided confidence level
        is ``1 - (1 - confidence) * 2``. Defaults to ``None``.
    **kwargs
        n_replicates : int
            Number of bootstrap resamples. Defaults to ``100``.
    """
    print("{}\n\n======".format(metric_field))
    df_stats = get_df_stats(data, metric_field)
    print("\n", df_stats, "\n")

    if threshold is not None:
        effective_confidence = 1 - (1 - confidence) * 2
        print(
            "One sided test: confidence level tested is: 1 - (1 - confidence)*2 = {}\n".format(
                effective_confidence
            )
        )
    else:
        effective_confidence = confidence

    get_mann_whitney_test(data, metric_field, effective_confidence)

    n_replicates = kwargs.get("n_replicates", 100)
    get_ci_bootstrap(
        data,
        metric_field,
        confidence=effective_confidence,
        n_replicates=n_replicates,
        plot=True,
        non_inferiority_threshold=threshold,
    )


def get_results(
    data: pd.DataFrame,
    metric_field: str,
    confidence: float = 0.9,
    threshold: Optional[float] = None,
    calculate_ratio: bool = True,
    **kwargs,
) -> None:
    """Full A/B experiment analysis pipeline.

    Computes summary statistics, runs the significance test, and renders a
    confidence-interval chart.

    Parameters
    ----------
    data : pd.DataFrame
        Experiment DataFrame with ``variant`` (0 = control, 1 = treatment)
        and ``metric_field`` columns.
    metric_field : str
        Metric to analyse.
    confidence : float
        Two-sided confidence level (e.g. ``0.9`` for 90 %). Defaults to ``0.9``.
    threshold : float, optional
        Non-inferiority margin (relative fraction). When provided the analysis
        is one-sided and the confidence level is adjusted to
        ``1 - (1 - confidence) * 2``. Defaults to ``None`` (two-sided).
    calculate_ratio : bool
        For continuous metrics, whether to report a relative (ratio) CI.
        Defaults to ``True``.
    **kwargs
        mannwhitney : bool
            If ``True``, also run the Mann-Whitney U test. Defaults to ``False``.
    """
    print("{}\n\n======".format(metric_field))
    df_stats = get_df_stats(data, metric_field)
    get_stats(df_stats, non_inferiority_threshold=threshold)
    print("\n", df_stats, "\n")

    if threshold is not None:
        effective_confidence = 1 - (1 - confidence) * 2
        print(
            "One sided test: confidence level tested is: 1 - (1 - confidence)*2 = {}\n".format(
                effective_confidence
            )
        )
    else:
        effective_confidence = confidence

    if kwargs.get("mannwhitney"):
        get_mann_whitney_test(data, metric_field, effective_confidence)

    get_ci(
        df_stats,
        metric_field,
        confidence=effective_confidence,
        plot=True,
        non_inferiority_threshold=threshold,
        ratio=calculate_ratio,
    )


def improv_interval_binomial(
    confidence: float,
    successes_base: float,
    successes_var: float,
    obs_base: int,
    obs_var: int,
) -> tuple:
    """Compute a Fieller CI for the relative improvement between two binomial proportions.

    Derives per-group proportions and their Bernoulli standard deviations,
    then delegates to :func:`confidence_interval_ratio`.

    Parameters
    ----------
    confidence : float
        Confidence level (e.g. ``0.9`` for 90 %).
    successes_base : float
        Number of conversions in the control group.
    successes_var : float
        Number of conversions in the treatment group.
    obs_base : int
        Total observations in the control group.
    obs_var : int
        Total observations in the treatment group.

    Returns
    -------
    tuple[float, float, float]
        ``(estimate, ci_l, ci_h)`` where
        ``estimate = (p_var - p_base) / p_base``.
        Returns ``(np.nan, np.nan, np.nan)`` for invalid inputs.
    """
    if not obs_base or not obs_var or successes_base > obs_base or successes_var > obs_var:
        return np.nan, np.nan, np.nan

    avg_base = successes_base / obs_base
    avg_var = successes_var / obs_var
    stdev_base = np.sqrt(avg_base * (1 - avg_base))
    stdev_var = np.sqrt(avg_var * (1 - avg_var))

    return confidence_interval_ratio(avg_base, stdev_base, obs_base, avg_var, stdev_var, obs_var, confidence)


def confidence_interval_ratio(
    avg_base: float,
    stdev_base: float,
    obs_base: int,
    avg_var: float,
    stdev_var: float,
    obs_var: int,
    confidence: float = 0.9,
) -> tuple:
    """Compute a Fieller CI for the relative difference ``(avg_var - avg_base) / |avg_base|``.

    Fieller's theorem yields an exact CI for a ratio of normally distributed
    random variables. The result is expressed on the relative scale
    ``(avg_var / avg_base) - 1``.

    Parameters
    ----------
    avg_base : float
        Mean of the control group.
    stdev_base : float
        Standard deviation of the control group.
    obs_base : int
        Number of observations in the control group.
    avg_var : float
        Mean of the treatment group.
    stdev_var : float
        Standard deviation of the treatment group.
    obs_var : int
        Number of observations in the treatment group.
    confidence : float
        Confidence level (e.g. ``0.9`` for 90 %). Defaults to ``0.9``.

    Returns
    -------
    tuple[float, float, float]
        ``(estimate, ci_l, ci_h)``.

        - Returns ``(estimate, None, None)`` when standard deviations or
          sample sizes are missing.
        - Returns ``(estimate, np.nan, np.nan)`` when Fieller's discriminant
          is non-positive (CI is undefined).
    """
    if not avg_base or avg_var is None or np.isnan(avg_var):
        return None, None, None

    alpha = 1 - confidence
    estimate = (avg_var - avg_base) / abs(avg_base)

    if (
        stdev_base is None or stdev_var is None
        or np.isnan(stdev_base) or np.isnan(stdev_var)
        or not obs_base or not obs_var
    ):
        return estimate, None, None

    zscore = norm.ppf(1 - alpha / 2)

    # Fieller's theorem components
    A = avg_base * avg_var
    B = avg_base**2 - (zscore**2 * stdev_base**2 / obs_base)
    C = avg_var**2 - (zscore**2 * stdev_var**2 / obs_var)
    discriminant = A**2 - B * C

    if discriminant <= 0 or B <= 0:
        return estimate, np.nan, np.nan

    half_width = np.sqrt(discriminant) / B
    fieller = [A / B - half_width - 1, A / B + half_width - 1]

    if avg_base < 0:
        fieller = [-x for x in fieller]

    return estimate, fieller[0], fieller[1]


def calculate_req_traffic_for_power(
    alpha: float,
    beta: float,
    base_rate: float,
    expected_effect: float,
) -> int:
    """Compute the required per-variant sample size for a two-proportion z-test.

    Uses the standard formula:

    .. math::

        n = \\frac{2\\,\\sigma^2\\,(z_{\\alpha/2} + z_{\\beta})^2}{\\delta^2}

    where :math:`\\sigma^2 = p(1-p)` and
    :math:`\\delta = p \\cdot \\text{expected\\_effect}`.

    Parameters
    ----------
    alpha : float
        Desired Type I error rate (e.g. ``0.05``).
    beta : float
        Desired Type II error rate (e.g. ``0.2`` for 80 % power).
    base_rate : float
        Baseline conversion rate (proportion).
    expected_effect : float
        Minimum detectable effect as a relative fraction of ``base_rate``
        (e.g. ``0.05`` for a 5 % relative lift).

    Returns
    -------
    int
        Required number of observations per variant.
    """
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(1 - beta)
    mde = base_rate * expected_effect
    sigma = np.sqrt(base_rate * (1 - base_rate))
    n_variant = 2 * (sigma**2) * (za + zb) ** 2 / (mde**2)
    return int(n_variant)


def calculate_mde_from_traffic(
    alpha: float,
    beta: float,
    base_rate: float,
    n_variant: int,
) -> float:
    """Compute the minimum detectable effect (MDE) given available traffic.

    Inverts the sample-size formula from :func:`calculate_req_traffic_for_power`
    to find the smallest relative effect detectable at the specified power.

    Parameters
    ----------
    alpha : float
        Desired Type I error rate (e.g. ``0.05``).
    beta : float
        Desired Type II error rate (e.g. ``0.2`` for 80 % power).
    base_rate : float
        Baseline conversion rate (proportion).
    n_variant : int
        Available number of observations per variant.

    Returns
    -------
    float
        MDE expressed as a relative fraction of ``base_rate``
        (e.g. ``0.05`` means a 5 % relative lift is detectable).
    """
    za = norm.ppf(1 - alpha / 2)
    zb = norm.ppf(1 - beta)
    sigma = np.sqrt(base_rate * (1 - base_rate))
    mde = np.sqrt(2 * (sigma**2) * (za + zb) ** 2 / n_variant)

    print(
        f"MDE (percentage effect) = {mde / base_rate:.2%}",
        f"\nMDE (absolute effect) = {base_rate:.2%} +- {mde:.2%}",
        f"\nEffect identifiable outside the range [{base_rate - mde:.2%},{base_rate + mde:.2%}]\n------",
    )
    return mde / base_rate
