import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
from scipy.stats import chi2_contingency, chi2, ttest_ind_from_stats, norm, binom, mannwhitneyu, t, chisquare
from statsmodels.stats.proportion import proportion_confint 
import statsmodels.formula.api as smf
# import pyspark.sql.functions as f

import patsy
import plotly.graph_objects as go


def vertical_plot_lines(x, xal, yal, **kwargs):
    plt.axvline(x.mean(), linestyle = '--', color = kwargs.get("color", "g"), alpha = 0.5)
    tx_mean = "mean: {:.3f}".format(x.mean())
    txkw = dict(size=11, color = kwargs.get("color", "g"), rotation=90)
    plt.text(x.mean()+xal, yal, tx_mean, **txkw)


def plot_distribution(df, metric, **kwargs):
    g = sns.FacetGrid(
        data=df,
        aspect=3, height = 5,
        hue = 'variant', row = kwargs.get('row',None)
    ) 
    g.map(sns.distplot, metric, hist=kwargs.get('displot_hist',False), kde_kws={'bw': kwargs.get('kde_bandwidth','silverman')})
    g.map(vertical_plot_lines, metric, xal=  kwargs.get('xal',0.10), yal=kwargs.get('yal',0.5))
    g.fig.suptitle("Distribution between variants - {}".format(metric))
    g.set_xlabels("{}".format(metric))
    g.set_ylabels("density")
    g.set(xlim = kwargs.get('xlim', (-10,10)))
    g.add_legend()
    plt.subplots_adjust(top=0.9)
    for ax in g.axes:
        ax[0].axvline(x=0, color="black", ls=':')
                
    
def one_sided_t_test(estimate_base,
                     sd_base, 
                     n_base, 
                     estimate_variant, 
                     sd_variant, 
                     n_variant,
                     acceptable_cost_value,
                     acceptable_cost_type='relative',
                     increase_is='good'):
    
    #calculate delta as absolute difference
    if acceptable_cost_type=='relative':
        delta = acceptable_cost_value*estimate_base
    elif acceptable_cost_type=='impact':
        delta = acceptable_cost_value/(n_variant+n_base)
        
    #determine direction of one sided test 
    if increase_is == 'good':
        threshold_value = estimate_base - delta
    elif increase_is == 'bad':
        threshold_value = estimate_base + delta
        
    # t_test 
    t_test = ttest_ind_from_stats(
                        mean1=threshold_value,
                        std1=sd_base, 
                        nobs1=n_base, 
                        mean2=estimate_variant, 
                        std2=sd_variant, 
                        nobs2=n_variant,
                        equal_var=False)
    #p_value one-sided instead of two sided
    p_value=t_test[1]/2.0 
    
    return p_value
 

def get_g_test(counts_base, counts_var, visitors_base, visitors_var):
    try:
        p_value = chi2_contingency(
            [[counts_base, counts_var], [visitors_base - counts_base, visitors_var - counts_var]], 
            correction = False, lambda_='log-likelihood'
        )[1]
    except ValueError:
        p_value = [nan]
    return p_value


def confidence_interval_mean_differences(
    confidence,
    avg_base, avg_var, 
    stdev_base, stdev_var, 
    obs_base, obs_var):
    
    pooled_se = np.sqrt(stdev_base**2 / obs_base + stdev_var**2 / obs_var)
    delta = avg_var - avg_base
    
    tstat = delta /  pooled_se
    df = (stdev_base**2 / obs_base + stdev_var**2 / obs_var)**2 / ((stdev_base**2)**2 / (obs_base**2 * (obs_base - 1)) + (stdev_var**2)**2 / (obs_var**2 * (obs_var - 1)))
    
    # two side t-test
    # p_val = 2 * t.cdf(-abs(tstat), df)
    
    # upper and lower bounds
    ci_l = delta - t.ppf(1-(1-confidence)/2, df)*pooled_se 
    ci_h = delta + t.ppf(1-(1-confidence)/2, df)*pooled_se 
    
    return delta, ci_l, ci_h

    
def plot_ci(estimate, ci_l, ci_h, metric_field, non_inferioirty_threshold=None, ratio = True):
    
    if non_inferioirty_threshold is None:
        significance_threshold = 0
    else: 
        significance_threshold = non_inferioirty_threshold
        
    if ((ci_l < significance_threshold) & (ci_h > significance_threshold)):
        stat_text='Effect is not statistically significant'
        if estimate > significance_threshold:
            color_marker = 'rgb(70,130,180)' # 'rgb(0,100,0)'
            color_ci = 'rgb(161,172,216)' # 'rgb(204,225,200)'
        else: 
            color_marker = 'rgb(70,130,180)' #'rgb(128,0,0)'
            color_ci = 'rgb(161,172,216)' # 'rgb(229,181,181)'
    else: 
        stat_text='Effect is statistically significant'
        if estimate > significance_threshold:
            color_marker = 'rgb(0,100,0)'
            color_ci = 'rgb(60,179,113)'
        else: 
            color_marker = 'rgb(128,0,0)'
            color_ci = 'rgb(250,128,114)'

    fig = go.Figure()
    if non_inferioirty_threshold is None:
        fig.add_trace(go.Scatter(y=[0], x=[estimate], 
                                 mode='markers', marker_symbol='line-ns',
                                 error_x=dict(type='data', symmetric=False, 
                                              array = [ci_h - estimate], arrayminus=[estimate - ci_l], 
                                              color=color_ci, thickness=100, width=0),
            hovertext=[stat_text],
            marker=dict(size=70, line_width=2, line_color=color_marker)
        ))
    else:
        fig.add_trace(go.Scatter(y=[0], x=[estimate], 
                                 mode='markers', marker_symbol='line-ns',
                                 error_x=dict(type='data', symmetric=False, 
                                              array = [999], arrayminus=[estimate - ci_l], 
                                              color=color_ci, thickness=100, width=0),
            opacity=1.0,
            hovertext=[stat_text],
            marker=dict(size=70, line_width=2, line_color=color_marker)
        ))
#         fig.add_shape(
#             dict(type="line", x0=(ci_h), y0=-0.45, x1=(ci_h), y1=0.45,
#                  line=dict(color="white", width=1.5, dash="dot")
#             ))
#         fig.add_shape(
#             dict(type="line", x0=(ci_l), y0=-0.45, x1=(ci_l), y1=0.45,
#                  line=dict(color="white", width=1.5, dash="dot")
#             ))
        
    fig.add_shape(
            dict(type="line", x0=0, y0=-1, x1=0, y1=1,
                 line=dict(color="black", width=2.5, dash="dot")
            ))
    
    if non_inferioirty_threshold is not None:
        fig.add_shape(
            dict(type="line", 
                 x0 = non_inferioirty_threshold, y0=-1, 
                 x1 = non_inferioirty_threshold, y1=1,
                 line=dict(color="black", width=4)
            ))

    fig.update_layout(
        title=metric_field, autosize=False, width=1000, height=400, 
        annotations=[dict(x=estimate, y=-0.65, text=stat_text, 
                          showarrow = False, xanchor="left", xshift=10, opacity=0.7, font=dict(size=15))],
        font=dict( family="Courier New, monospace", size=18, color="#7f7f7f")
    )
    if ratio == False:
        fig['layout']['annotations'] += tuple(
            [dict(x=estimate, y=-0.78, text='Axis units are in absolute values',
                  showarrow = False, xanchor="left", xshift=10, opacity=0.7, font=dict(size=12))])
        
    fig.update_yaxes(showticklabels=False)
    if non_inferioirty_threshold is not None:
        fig.update_xaxes(dict(range=[min(-0.3,non_inferioirty_threshold),0.3]))
    else:
        fig.update_xaxes(dict(range=[-0.3,0.3]))
        
    if ((ci_l < -0.3) | (ci_h > 0.3)):
        fig.update_xaxes(dict(range=[min(ci_l - 0.05, - 0.05), max(ci_h + 0.05, 0.05)]))
        
    fig.show()
    

def get_ci(df_stats, metric_field, confidence=0.9, plot = True, non_inferioirty_threshold=None, ratio=True):
    
    if df_stats['binary'].all() == True:
        estimate, ci_l, ci_h = improv_interval_binomial(
            confidence = confidence, 
            successes_base=df_stats.at[0, 'reached_goal'], 
            successes_var=df_stats.at[1, 'reached_goal'], 
            obs_base=df_stats.at[0, 'visitors'], 
            obs_var=df_stats.at[1, 'visitors'])
        print("Estimate: {:.3%},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))
    else:     
        if ratio:
            estimate, ci_l, ci_h = confidence_interval_ratio(
                confidence = confidence, 
                avg_base = df_stats.at[0, 'reached_goal']/df_stats.at[0, 'visitors'], 
                avg_var = df_stats.at[1, 'reached_goal']/df_stats.at[1, 'visitors'], 
                stdev_base = df_stats.at[0, 'stdv'],  
                stdev_var = df_stats.at[1, 'stdv'], 
                obs_base=df_stats.at[0, 'visitors'], 
                obs_var=df_stats.at[1, 'visitors'])
            print("Estimate: {:.3%},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))
        else:
            estimate, ci_l, ci_h = confidence_interval_mean_differences(
                confidence = confidence, 
                avg_base = df_stats.at[0, 'reached_goal']/df_stats.at[0, 'visitors'], 
                avg_var = df_stats.at[1, 'reached_goal']/df_stats.at[1, 'visitors'], 
                stdev_base = df_stats.at[0, 'stdv'],  
                stdev_var = df_stats.at[1, 'stdv'], 
                obs_base=df_stats.at[0, 'visitors'], 
                obs_var=df_stats.at[1, 'visitors'])
            print("Estimate: {:.3f},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))
    
    if ci_h < ci_l:
        print("*** ci_h and ci_l are reversed - check why ***")
        ci_l, ci_h = min(ci_l,ci_h), max(ci_l,ci_h)
    
    if (non_inferioirty_threshold is not None) & ratio:
        print("CI_low > {:.0%}? {}".format(non_inferioirty_threshold, ci_l > non_inferioirty_threshold))
    elif (non_inferioirty_threshold is not None) & (not ratio):
        print("CI_low > {:.2f}? {}".format(non_inferioirty_threshold, ci_l > non_inferioirty_threshold))
    if plot:
        plot_ci(estimate, ci_l, ci_h, metric_field, non_inferioirty_threshold=non_inferioirty_threshold, ratio=ratio)
        

def get_mann_whitney_test(data, metric_field, confidence=0.9):
    array_base = data.loc[data['variant'] == 0, metric_field]
    array_variant = data.loc[data['variant'] == 1, metric_field]
    u_value, p_value = mannwhitneyu(array_base, array_variant)
    
    print("Mann-Whitney p-value: {:.5f} \nstatstistical significance at {} level p_value < (1 - confidence): {}\n".format(
        p_value, confidence, p_value<(1-confidence)))
    
    return u_value, p_value


def get_ci_bootstrap(data, metric_field, confidence=0.9, n_replicates=100, plot=True, non_inferioirty_threshold=None):
    print("""\n=== Bootstrap sampling with {} replications to get non-paramteric CI ===""".format(n_replicates))
    bootstraps = pd.DataFrame()
    for i in range(n_replicates):
        df_bootstrap = data[['variant',metric_field]].sample(len(data), replace=True).groupby('variant').mean().reset_index()
        bootstraps = bootstraps.append(df_bootstrap)

        bootstraps_df = pd.DataFrame(bootstraps)

    df_stats = get_df_stats(bootstraps_df, metric_field)
    
    bs_mean_diff = (
        np.array(bootstraps_df[bootstraps_df.variant == 1][metric_field]) 
        - np.array(bootstraps_df[bootstraps_df.variant == 0][metric_field])
    )
    
    estimate = bs_mean_diff.mean()
    ci_l = np.percentile(bs_mean_diff, 100*(1 - confidence)/2)
    ci_h = np.percentile(bs_mean_diff, 100*(1 - (1 - confidence)/2))
    print("Estimate: {:.3f},     CI = [{:.5f}, {:.5f}]".format(estimate, ci_l, ci_h))
    if (non_inferioirty_threshold is not None):
        print("CI_low > {:.2f}? {}".format(non_inferioirty_threshold, ci_l > non_inferioirty_threshold))

    plot_ci(estimate, ci_l, ci_h, metric_field,  
        non_inferioirty_threshold=non_inferioirty_threshold,
        ratio = False
           )
    
    return bootstraps_df

      
def get_stats(df_stats, non_inferioirity_threshold=None):
    
    if non_inferioirity_threshold is None:
        two_sided = True
    else: 
        two_sided = False 
        
        
    if df_stats['binary'].all() == True:
        p_val = get_g_test(counts_base = df_stats.at[0, 'reached_goal'],
               counts_var = df_stats.at[1, 'reached_goal'],
               visitors_base = df_stats.at[0, 'visitors'],
               visitors_var = df_stats.at[1, 'visitors']
              )
        print('\npval = {:.5f}, significant at 10%: {}\n'.format(p_val, p_val<0.1))
    else:
        if two_sided:
            p_val = ttest_ind_from_stats(
                mean1=df_stats['reached_goal'][0]/df_stats['visitors'][0],
                std1=df_stats['stdv'][0], 
                nobs1=df_stats['visitors'][0], 
                mean2=df_stats['reached_goal'][1]/df_stats['visitors'][1],
                std2=df_stats['stdv'][1], 
                nobs2=df_stats['visitors'][1],
                equal_var=False
            )[1]
            print('\npval = {:.5f}, significant at 10%: {}\n'.format(p_val, p_val<0.1))
        else:
            p_val = ttest_ind_from_stats(
                mean1=df_stats['reached_goal'][0]/df_stats['visitors'][0] * (1 + non_inferioirity_threshold),
                std1=df_stats['stdv'][0], 
                nobs1=df_stats['visitors'][0], 
                mean2=df_stats['reached_goal'][1]/df_stats['visitors'][1],
                std2=df_stats['stdv'][1], 
                nobs2=df_stats['visitors'][1],
                equal_var=False
            )[1]/2 
            print('\nNon-inferioirty threshold: {:.0%}, pval = {:.5f}, significant at 10%: {}\n'.format(non_inferioirity_threshold, p_val, p_val<0.1))
        
    avg_base = df_stats['reached_goal'][0]/df_stats['visitors'][0]
    avg_var = df_stats['reached_goal'][1]/df_stats['visitors'][1]
        
    response = {
        "p-value": p_val,
        "N": df_stats.at[0, 'visitors'] + df_stats.at[1, 'visitors'],
        "ratio var/base": (avg_var/avg_base)-1
    }
    
    return response

        
def get_df_stats(df, metric_field):
    df_stats = df.groupby('variant').agg({metric_field: ['count', 'sum','mean','std']}).reset_index()
    df_stats.columns = ['variant','visitors','reached_goal','average_value','stdv']
    if np.isin(df[metric_field].unique(), [0,1]).all():
        df_stats['binary'] = True
    else: 
        df_stats['binary'] = False
    
    return df_stats


def get_results_per_clienttype(data, metric_field, confidence=0.9, threshold=None, calculate_ratio=True):
    print("{}\n\n======".format(metric_field))
    for ctg in ['web|app','web','app']:
        print("\n"+ctg+"\n------")
        data_ctg = data[data['clienttype_grouped'].str.contains(ctg)]

        df_stats = get_df_stats(data_ctg, metric_field)
        get_stats(df_stats, non_inferioirity_threshold=threshold)
        print("\n", df_stats, "\n")
        
        if (threshold is not None):
            confidence = 1 - (1 - confidence)*2
            print("One sided test: confidence level tested is: 1 - (1 - confidence)*2 = {}\n".format(confidence))

        get_ci(
            df_stats,
            metric_field,
            confidence=confidence, 
            plot = True, 
            non_inferioirty_threshold=threshold,
            ratio = calculate_ratio
        )
        
        
def get_results_bootstrap(data, metric_field, confidence=0.9, threshold=None, **kwargs):
    print("{}\n\n======".format(metric_field))
    df_stats = get_df_stats(data, metric_field)
    print("\n", df_stats, "\n")
    
    if (threshold is not None):
        confidence = 1 - (1 - confidence)*2
        print("One sided test: confidence level tested is: 1 - (1 - confidence)*2 = {}\n".format(confidence))
    
    get_mann_whitney_test(data, metric_field, confidence)

    if kwargs.get('n_replicates') is not None:
        n_replicates = kwargs.get('n_replicates')
    else: 
        n_replicates = 100
        
    get_ci_bootstrap(
        data, 
        metric_field, 
        confidence=confidence, 
        n_replicates=n_replicates, 
        plot=True,
        non_inferioirty_threshold=threshold
    )
    
        
def get_results(data, metric_field, confidence=0.9, threshold=None, calculate_ratio=True, **kwargs):
    print("{}\n\n======".format(metric_field))
    df_stats = get_df_stats(data, metric_field)
    get_stats(df_stats, non_inferioirity_threshold=threshold)
    print("\n", df_stats, "\n")
    
    if (threshold is not None):
        confidence = 1 - (1 - confidence)*2
        print("One sided test: confidence level tested is: 1 - (1 - confidence)*2 = {}\n".format(confidence))
        
    if kwargs.get('mannwhitney'):
        get_mann_whitney_test(data, metric_field, confidence)

    get_ci(
        df_stats,
        metric_field,
        confidence=confidence, 
        plot = True, 
        non_inferioirty_threshold=threshold,
        ratio = calculate_ratio
    )
    
def improv_interval_binomial(confidence, successes_base, successes_var, obs_base, obs_var):
    if (not obs_base
        or not obs_var
        or (successes_base > obs_base)
        or (successes_var > obs_var) ):
        return np.nan, np.nan, np.nan

    avg_base = successes_base/obs_base
    avg_var = successes_var/obs_var
    stdev_base = np.sqrt((avg_base-avg_base**2))
    stdev_var = np.sqrt((avg_var-avg_var**2))

    return confidence_interval_ratio(avg_base, stdev_base, obs_base, avg_var, stdev_var, obs_var, confidence)

def calculate_req_traffic_for_power(alpha, beta, base_rate, expected_effect):
    za = norm.ppf(1 - alpha/2)
    zb = norm.ppf(1 - beta)
    mde = (base_rate * (1+expected_effect) - base_rate) / np.sqrt((base_rate * (1+expected_effect)) * (1-base_rate * (1+expected_effect)))
    req_n = 2 * (za + zb)**2 / (mde)**2
    return int(2 * req_n)

    
def confidence_interval_ratio(avg_base, stdev_base, obs_base, avg_var, stdev_var, obs_var, confidence=0.9):
    if (not avg_base or
            avg_var is None or np.isnan(avg_var)):
        return None, None, None
    
    alpha = 1-confidence

    estimate = (avg_var - avg_base) / abs(avg_base)

    if (stdev_base is None or stdev_var is None or 
        np.isnan(stdev_base) or np.isnan(stdev_var) or
        not obs_base or not obs_var):
        return estimate, None, None

    zscore = norm.ppf(1 - alpha/2)
    A = avg_base*avg_var
    B = avg_base**2 - (zscore**2 * stdev_base**2/obs_base)
    C = avg_var**2 - (zscore**2 * stdev_var**2/obs_var)
    sqrt = A**2 - (B*C)

    if (sqrt <= 0 or B <= 0):
        return estimate, np.nan, np.nan

    range = np.sqrt(sqrt)/B
    fieller = [A/B - range - 1, A/B + range - 1]

    if (avg_base < 0):
        fieller = [-x for x in fieller]
    return estimate, fieller[0], fieller[1]


def calculate_req_traffic_for_power(alpha, beta, base_rate, expected_effect):
    za = norm.ppf(1 - alpha/2)
    zb = norm.ppf(1 - beta)
    mde = (base_rate * expected_effect) 
    sigma = np.sqrt((base_rate) * (1-base_rate))
    n_variant = 2 * (sigma**2) * (za + zb)**2 / (mde**2)
    return int(n_variant)


def calculate_mde_from_traffic(alpha, beta, base_rate, n_variant):
    za = norm.ppf(1 - alpha/2)
    zb = norm.ppf(1 - beta)
    sigma = np.sqrt((base_rate) * (1-base_rate))    
    mde = np.sqrt(2 * (sigma**2) * (za + zb)**2 / n_variant)
    
    print(
        f'MDE (percentage effect) = {mde/base_rate:.2%}',
        f'\nMDE (absolute effect) = {base_rate:.2%} +- {mde:.2%}',
        f'\nEffect identifiable outside the range [{base_rate - mde:.2%},{base_rate + mde:.2%}]\n------'
    )
    return mde / BASE_RATE
