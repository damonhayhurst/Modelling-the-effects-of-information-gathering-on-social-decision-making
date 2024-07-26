

import itertools
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, concat
from scipy.stats import ttest_ind, false_discovery_control
from utils.display import display
from utils.masks import get_positive_gain_of_less_than_ten
from utils.columns import CLUSTER, CLUSTER_1, CLUSTER_2, GAIN_OF_TEN, GAIN_UNDER_TEN, LIE, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, OTHER_LIE, OTHER_TRUTH, P_VALUE, SELECTED_AOI, SELF_LIE, SELF_TRUE, T_STATISTIC, TRIAL_COUNT
from utils.masks import get_gain_of_ten
from statsmodels.stats.multitest import multipletests
from scipy.stats._result_classes import TtestResult


def get_test(cluster_df: DataFrame, analysis_df: DataFrame, index: Index):
    cluster_trials = analysis_df.loc[index.isin(cluster_df.index)]
    result = cluster_trials[SELECTED_AOI] == LIE
    return result


def get_is_lie(df: DataFrame):
    return df[SELECTED_AOI] == LIE


def get_cluster_trials(cluster_df: DataFrame, analysis_df: DataFrame, index: Index):
    return analysis_df.loc[index.isin(cluster_df.index)]


def percent_lie(df: DataFrame):
    is_lie = get_is_lie(df)
    return is_lie.mean() * 100, is_lie.std() * 100


def n_transitions(df: DataFrame):
    mean_transitions = (df[N_TRANSITIONS]).mean()
    std_transitions = (df[N_TRANSITIONS]).std()
    return mean_transitions, std_transitions


def avg_dwell_time(df: DataFrame, aoi: str):
    return df[aoi].mean(), df[aoi].std()


def n_trials(df: DataFrame):
    return len(df)


def gain_of_ten(df: DataFrame):
    is_gain_of_ten = get_gain_of_ten(df)
    return is_gain_of_ten.mean() * 100, is_gain_of_ten.std() * 100


def gain_under_ten(df: DataFrame):
    is_gain_under_ten = get_positive_gain_of_less_than_ten(df)
    return is_gain_under_ten.mean() * 100, is_gain_under_ten.std() * 100


def calc_percent_error_bounds(df: DataFrame):
    val, err = df
    upper_err, lower_err = min(val + err/2, 100), max(val - err/2, 0)
    return (val, (lower_err, upper_err))


def calc_duration_error_bounds(df: DataFrame):
    val, err = df
    upper_err, lower_err = val + err/2, max(val - err/2, 0)
    return (val, (lower_err, upper_err))


def calc_count_error_bounds(df: DataFrame):
    val, err = df[0], df[1]
    upper_err, lower_err = val + err/2, max(val - err/2, 0)
    upper_err = upper_err if np.isnan(upper_err) else int(upper_err)
    lower_err = lower_err if np.isnan(lower_err) else int(lower_err)
    return (val, (lower_err, upper_err))


def no_error_bounds(df: DataFrame):
    return (df, (np.nan, np.nan))


def get_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame, index: Index):

    cluster_trials: DataFrame = cluster_df.apply(lambda cluster: get_cluster_trials(cluster, analysis_df, index)).groupby(CLUSTER)

    return DataFrame({
        LIE: cluster_trials.apply(percent_lie).apply(calc_percent_error_bounds),
        N_TRANSITIONS: cluster_trials.apply(n_transitions).apply(calc_count_error_bounds),
        SELF_LIE: cluster_trials.apply(avg_dwell_time, aoi=SELF_LIE).apply(calc_duration_error_bounds),
        SELF_TRUE: cluster_trials.apply(avg_dwell_time, aoi=SELF_TRUE).apply(calc_duration_error_bounds),
        OTHER_LIE: cluster_trials.apply(avg_dwell_time, aoi=OTHER_LIE).apply(calc_duration_error_bounds),
        OTHER_TRUTH: cluster_trials.apply(avg_dwell_time, aoi=OTHER_TRUTH).apply(calc_duration_error_bounds),
        TRIAL_COUNT: cluster_trials.apply(n_trials).apply(no_error_bounds),
        GAIN_OF_TEN: cluster_trials.apply(gain_of_ten).apply(calc_percent_error_bounds),
        GAIN_UNDER_TEN: cluster_trials.apply(gain_under_ten).apply(calc_percent_error_bounds),
    })


def get_response_stats_for_group_by(group_by_df: DataFrame):

    return DataFrame({
        LIE: group_by_df.apply(percent_lie).apply(calc_percent_error_bounds),
        N_TRANSITIONS: group_by_df.apply(n_transitions).apply(calc_count_error_bounds),
        SELF_LIE: group_by_df.apply(avg_dwell_time, aoi=SELF_LIE).apply(calc_duration_error_bounds),
        SELF_TRUE: group_by_df.apply(avg_dwell_time, aoi=SELF_TRUE).apply(calc_duration_error_bounds),
        OTHER_LIE: group_by_df.apply(avg_dwell_time, aoi=OTHER_LIE).apply(calc_duration_error_bounds),
        OTHER_TRUTH: group_by_df.apply(avg_dwell_time, aoi=OTHER_TRUTH).apply(calc_duration_error_bounds),
        TRIAL_COUNT: group_by_df.apply(n_trials).apply(no_error_bounds),
        GAIN_OF_TEN: group_by_df.apply(gain_of_ten).apply(calc_percent_error_bounds),
        GAIN_UNDER_TEN: group_by_df.apply(gain_under_ten).apply(calc_percent_error_bounds),
    })


def get_response_stats(analysis_df: DataFrame):

    return Series({
        LIE: calc_percent_error_bounds(percent_lie(analysis_df)),
        N_TRANSITIONS: calc_count_error_bounds(n_transitions(analysis_df)),
        SELF_LIE: calc_duration_error_bounds(avg_dwell_time(analysis_df, aoi=SELF_LIE)),
        SELF_TRUE: calc_duration_error_bounds(avg_dwell_time(analysis_df, aoi=SELF_TRUE)),
        OTHER_LIE: calc_duration_error_bounds(avg_dwell_time(analysis_df, aoi=OTHER_LIE)),
        OTHER_TRUTH: calc_duration_error_bounds(avg_dwell_time(analysis_df, aoi=OTHER_TRUTH)),
        TRIAL_COUNT: no_error_bounds(n_trials(analysis_df)),
        GAIN_OF_TEN: calc_percent_error_bounds(gain_of_ten(analysis_df)),
        GAIN_UNDER_TEN: calc_percent_error_bounds(gain_under_ten(analysis_df))
    })


def get_t_test_response_stats(a: DataFrame, b: DataFrame):

    return Series({
        LIE: ttest_ind(get_is_lie(a), get_is_lie(b)),
        N_TRANSITIONS: ttest_ind(a[N_TRANSITIONS], b[N_TRANSITIONS]),
        SELF_LIE: ttest_ind(a[SELF_LIE], b[SELF_LIE]),
        SELF_TRUE: ttest_ind(a[SELF_TRUE], b[SELF_TRUE]),
        OTHER_LIE: ttest_ind(a[OTHER_LIE], b[OTHER_LIE]),
        OTHER_TRUTH: ttest_ind(a[OTHER_TRUTH], b[OTHER_TRUTH])
    })


def get_combination_t_test_response_stats(group_by_df):

    groups = group_by_df.groups.keys()
    combs = list(itertools.combinations(groups, 2))
    t_tests = DataFrame(index=MultiIndex.from_tuples(combs), columns=[LIE, N_TRANSITIONS, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH])
    for group1, group2 in combs:
        df1, df2 = group_by_df.get_group(group1), group_by_df.get_group(group2)
        t_tests.loc[(group1, group2), [LIE, N_TRANSITIONS, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]] = get_t_test_response_stats(
            df1, df2)
        
    t_tests = correct_p_values(t_tests)

    return t_tests


def correct_p_values(t_tests_df: DataFrame):

    for column in t_tests_df.columns:
        p_values = t_tests_df[column].apply(lambda cell: cell[1]).to_numpy()
        adjusted = false_discovery_control(p_values)
        for i in range(len(t_tests_df[column])):
            old = t_tests_df.iloc[i][column]
            t_tests_df.iloc[i][column] = (old[0], adjusted[i])

    return t_tests_df


def get_aggregate_cluster_t_test_response_stats(t_test_df: DataFrame):

    cluster_values = set(t_test_df[CLUSTER_1].tolist() + t_test_df[CLUSTER_2].tolist())
    combs = list(itertools.combinations_with_replacement(cluster_values, 2))
    agg_t_test_df = DataFrame(index=MultiIndex.from_tuples(combs), columns=[LIE, N_TRANSITIONS, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH])
    for cluster1, cluster2 in combs:
        is_cluster_comb = ((t_test_df[CLUSTER_1] == cluster1) & (t_test_df[CLUSTER_2] == cluster2)) | ((t_test_df[CLUSTER_1] == cluster2) & (t_test_df[CLUSTER_2] == cluster1))
        is_cluster_comb_df = t_test_df.loc[is_cluster_comb]
        for col in [LIE, N_TRANSITIONS, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]:
            t = is_cluster_comb_df[col].apply(lambda x: x[0])
            p = is_cluster_comb_df[col].apply(lambda x: x[1])
            agg_t_test_df.loc[(cluster1, cluster2), col] = (t.mean(), p.mean())

    agg_t_test_df = correct_p_values(agg_t_test_df)

    return agg_t_test_df
