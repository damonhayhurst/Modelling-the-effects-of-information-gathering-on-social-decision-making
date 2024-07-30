

import itertools
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, concat
from scipy.stats import ttest_ind, false_discovery_control
from utils.display import display
from utils.masks import get_gain_of_less_than_ten, get_gain_of_thirty, get_gain_of_twenty, get_loss_of_less_than_ten, get_loss_of_ten, get_loss_of_thirty, get_negative_gain, get_positive_gain, get_positive_gain_of_less_than_ten
from utils.columns import CLUSTER, CLUSTER_1, CLUSTER_2, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_OF_TWENTY, GAIN_UNDER_TEN, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_OF_TWENTY, LOSS_UNDER_TEN, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, NEGATIVE_GAIN, OTHER_LIE, OTHER_TRUTH, P_VALUE, POSITIVE_GAIN, RT, SELECTED_AOI, SELF_LIE, SELF_TRUE, T_STATISTIC, TRIAL_COUNT
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
    is_gain_under_ten = get_gain_of_less_than_ten(df)
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
    upper_err = upper_err if np.isnan(upper_err) else upper_err
    lower_err = lower_err if np.isnan(lower_err) else lower_err
    return (val, (lower_err, upper_err))


def no_error_bounds(df: DataFrame):
    return (df, (np.nan, np.nan))


def get_calc_error_bounds_fn(calc_error_bounds: bool = True, val_only: bool = False):
    def no_calc(df): return df if not val_only or not isinstance(df, tuple) else df[0]
    return {
        LIE: calc_percent_error_bounds if calc_error_bounds else no_calc,
        N_TRANSITIONS: calc_count_error_bounds if calc_error_bounds else no_calc,
        SELF_LIE: calc_duration_error_bounds if calc_error_bounds else no_calc,
        SELF_TRUE: calc_duration_error_bounds if calc_error_bounds else no_calc,
        OTHER_LIE: calc_duration_error_bounds if calc_error_bounds else no_calc,
        OTHER_TRUTH: calc_duration_error_bounds if calc_error_bounds else no_calc,
        TRIAL_COUNT: no_error_bounds if calc_error_bounds else no_calc,
        GAIN_OF_TEN: calc_percent_error_bounds if calc_error_bounds else no_calc,
        GAIN_UNDER_TEN: calc_percent_error_bounds if calc_error_bounds else no_calc,
        RT: calc_duration_error_bounds if calc_error_bounds else no_calc
    }


def get_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame, index: Index, with_error_bounds: bool = True, val_only: bool = False):

    cluster_trials: DataFrame = cluster_df.apply(lambda cluster: get_cluster_trials(cluster, analysis_df, index)).groupby(CLUSTER)
    calc_error_bounds = get_calc_error_bounds_fn(with_error_bounds, val_only)

    return DataFrame({
        LIE: cluster_trials.apply(percent_lie).apply(calc_error_bounds[LIE]),
        N_TRANSITIONS: cluster_trials.apply(n_transitions).apply(calc_error_bounds[N_TRANSITIONS]),
        SELF_LIE: cluster_trials.apply(avg_dwell_time, aoi=SELF_LIE).apply(calc_error_bounds[SELF_LIE]),
        SELF_TRUE: cluster_trials.apply(avg_dwell_time, aoi=SELF_TRUE).apply(calc_error_bounds[SELF_TRUE]),
        OTHER_LIE: cluster_trials.apply(avg_dwell_time, aoi=OTHER_LIE).apply(calc_error_bounds[OTHER_LIE]),
        OTHER_TRUTH: cluster_trials.apply(avg_dwell_time, aoi=OTHER_TRUTH).apply(calc_error_bounds[OTHER_TRUTH]),
        TRIAL_COUNT: cluster_trials.apply(n_trials).apply(calc_error_bounds[TRIAL_COUNT]),
        GAIN_OF_TEN: cluster_trials.apply(gain_of_ten).apply(calc_error_bounds[GAIN_OF_TEN]),
        GAIN_UNDER_TEN: cluster_trials.apply(gain_under_ten).apply(calc_error_bounds[GAIN_UNDER_TEN]),
    })


def get_response_stats_for_group_by(group_by_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):

    calc_error_bounds = get_calc_error_bounds_fn(with_error_bounds, val_only)

    return DataFrame({
        LIE: group_by_df.apply(percent_lie).apply(calc_error_bounds[LIE]),
        N_TRANSITIONS: group_by_df.apply(n_transitions).apply(calc_error_bounds[N_TRANSITIONS]),
        SELF_LIE: group_by_df.apply(avg_dwell_time, aoi=SELF_LIE).apply(calc_error_bounds[SELF_LIE]),
        SELF_TRUE: group_by_df.apply(avg_dwell_time, aoi=SELF_TRUE).apply(calc_error_bounds[SELF_TRUE]),
        OTHER_LIE: group_by_df.apply(avg_dwell_time, aoi=OTHER_LIE).apply(calc_error_bounds[OTHER_LIE]),
        OTHER_TRUTH: group_by_df.apply(avg_dwell_time, aoi=OTHER_TRUTH).apply(calc_error_bounds[OTHER_TRUTH]),
        TRIAL_COUNT: group_by_df.apply(n_trials).apply(calc_error_bounds[TRIAL_COUNT]),
        GAIN_OF_TEN: group_by_df.apply(gain_of_ten).apply(calc_error_bounds[GAIN_OF_TEN]),
        GAIN_UNDER_TEN: group_by_df.apply(gain_under_ten).apply(calc_error_bounds[GAIN_UNDER_TEN]),
        RT: group_by_df.apply(avg_dwell_time, aoi=RT).apply(calc_error_bounds[RT])
    })


def get_response_stats(analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):

    calc_error_bounds = get_calc_error_bounds_fn(with_error_bounds, val_only)

    return Series({
        LIE: calc_error_bounds[LIE](percent_lie(analysis_df)),
        N_TRANSITIONS: calc_error_bounds[N_TRANSITIONS](n_transitions(analysis_df)),
        SELF_LIE: calc_error_bounds[SELF_LIE](avg_dwell_time(analysis_df, aoi=SELF_LIE)),
        SELF_TRUE: calc_error_bounds[SELF_TRUE](avg_dwell_time(analysis_df, aoi=SELF_TRUE)),
        OTHER_LIE: calc_error_bounds[OTHER_LIE](avg_dwell_time(analysis_df, aoi=OTHER_LIE)),
        OTHER_TRUTH: calc_error_bounds[OTHER_TRUTH](avg_dwell_time(analysis_df, aoi=OTHER_TRUTH)),
        TRIAL_COUNT: calc_error_bounds[TRIAL_COUNT](n_trials(analysis_df)),
        GAIN_OF_TEN: calc_error_bounds[GAIN_OF_TEN](gain_of_ten(analysis_df)),
        GAIN_UNDER_TEN: calc_error_bounds[GAIN_UNDER_TEN](gain_under_ten(analysis_df)),
        RT: calc_error_bounds[RT](avg_dwell_time(analysis_df, aoi=RT))
    })


def get_t_test_response_stats(a: DataFrame, b: DataFrame, equal_var: bool = False):

    return Series({
        LIE: ttest_ind(get_is_lie(a), get_is_lie(b), equal_var=equal_var),
        N_TRANSITIONS: ttest_ind(a[N_TRANSITIONS], b[N_TRANSITIONS], equal_var=equal_var),
        SELF_LIE: ttest_ind(a[SELF_LIE], b[SELF_LIE], equal_var=equal_var),
        SELF_TRUE: ttest_ind(a[SELF_TRUE], b[SELF_TRUE], equal_var=equal_var),
        OTHER_LIE: ttest_ind(a[OTHER_LIE], b[OTHER_LIE], equal_var=equal_var),
        OTHER_TRUTH: ttest_ind(a[OTHER_TRUTH], b[OTHER_TRUTH], equal_var=equal_var)
    })


def get_combination_t_test_response_stats(group_by_df):

    groups = group_by_df.groups.keys()
    combs = list(itertools.combinations(groups, 2))
    t_tests = DataFrame(index=MultiIndex.from_tuples(combs), columns=[LIE, N_TRANSITIONS, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH])
    for group1, group2 in combs:
        df1, df2 = group_by_df.get_group(group1), group_by_df.get_group(group2)
        df1, df2 = get_response_stats(df1, with_error_bounds=False, val_only=True), get_response_stats(df2, with_error_bounds=False, val_only=True)
        t_tests.loc[(group1, group2), [LIE, N_TRANSITIONS, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]] = get_t_test_response_stats(
            df1, df2)
    
    t_tests = correct_p_values(t_tests)

    return t_tests


def correct_p_values(t_tests_df: DataFrame):

    for column in t_tests_df.columns:
        p_values = t_tests_df[column].apply(lambda cell: cell[1]).to_numpy()
        p_values = np.nan_to_num(p_values, nan=0)
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


def get_gains_response_stats(aoi_analysis_df, gain_labels: list[str], with_error_bounds: bool = True, val_only: bool = False):
    gains = {}
    for gain in gain_labels:
        if gain == GAIN_OF_TEN:
            gain_of_ten_trials = aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_of_ten_trials, with_error_bounds, val_only)
            gains[GAIN_OF_TEN] = response_df
        if gain == GAIN_UNDER_TEN:
            gain_under_ten_trials = aoi_analysis_df.loc[get_gain_of_less_than_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_under_ten_trials, with_error_bounds, val_only)
            gains[GAIN_UNDER_TEN] = response_df
        if gain == GAIN_OF_TWENTY in gain_labels:
            gain_of_twenty_trials = aoi_analysis_df.loc[get_gain_of_twenty(aoi_analysis_df)]
            response_df = get_response_stats(gain_of_twenty_trials, with_error_bounds, val_only)
            gains[GAIN_OF_TWENTY] = response_df
        if gain == GAIN_OF_THIRTY:
            gain_of_thirty_trials = aoi_analysis_df.loc[get_gain_of_thirty(aoi_analysis_df)]
            response_df = get_response_stats(gain_of_thirty_trials, with_error_bounds, val_only)
            gains[GAIN_OF_THIRTY] = response_df
        if gain == NEGATIVE_GAIN:
            negative_gain_trials = aoi_analysis_df.loc[get_negative_gain(aoi_analysis_df)]
            response_df = get_response_stats(negative_gain_trials, with_error_bounds, val_only)
            gains[NEGATIVE_GAIN] = response_df
        if gain == POSITIVE_GAIN:
            positive_gain_trials = aoi_analysis_df.loc[get_positive_gain(aoi_analysis_df)]
            response_df = get_response_stats(positive_gain_trials, with_error_bounds, val_only)
            gains[POSITIVE_GAIN] = response_df
        if gain == LOSS_OF_TEN:
            loss_of_ten_trials = aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df)]
            response_df = get_response_stats(loss_of_ten_trials, with_error_bounds, val_only)
            gains[LOSS_OF_TEN] = response_df
        if gain == LOSS_UNDER_TEN:
            loss_under_ten = aoi_analysis_df.loc[get_loss_of_less_than_ten(aoi_analysis_df)]
            response_df = get_response_stats(loss_under_ten, with_error_bounds, val_only)
            gains[LOSS_UNDER_TEN] = response_df
        if gain == LOSS_OF_TWENTY:
            loss_of_twenty = aoi_analysis_df.loc[get_gain_of_twenty(aoi_analysis_df)]
            response_df = get_response_stats(loss_of_twenty, with_error_bounds, val_only)
            gains[LOSS_OF_TWENTY] = response_df
        if gain == LOSS_OF_THIRTY:
            loss_of_thirty = aoi_analysis_df.loc[get_loss_of_thirty(aoi_analysis_df)]
            response_df = get_response_stats(loss_of_thirty, with_error_bounds, val_only)
            gains[LOSS_OF_THIRTY] = response_df
        if gain == f"{GAIN_UNDER_TEN}, {LOSS_UNDER_TEN}":
            gain_and_loss_ten_trials = aoi_analysis_df.loc[get_gain_of_less_than_ten(aoi_analysis_df) & get_loss_of_less_than_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_and_loss_ten_trials, with_error_bounds, val_only)
            gains[f"{GAIN_UNDER_TEN}, {LOSS_UNDER_TEN}"] = response_df
        if gain == f"{GAIN_UNDER_TEN}, {LOSS_OF_TEN}":
            gain_under_and_loss_ten_trials = aoi_analysis_df.loc[get_gain_of_less_than_ten(aoi_analysis_df) & get_loss_of_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_under_and_loss_ten_trials, with_error_bounds, val_only)
            gains[f"{GAIN_UNDER_TEN}, {LOSS_OF_TEN}"] = response_df
        if gain == f"{GAIN_OF_TEN}, {LOSS_UNDER_TEN}":
            gain_and_loss_ten_trials = aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df) & get_loss_of_less_than_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_and_loss_ten_trials, with_error_bounds, val_only)
            gains[f"{GAIN_OF_TEN}, {LOSS_UNDER_TEN}"] = response_df
        if gain == f"{GAIN_OF_TEN}, {LOSS_OF_TEN}":
            gain_under_and_loss_ten_trials = aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df) & get_loss_of_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_under_and_loss_ten_trials, with_error_bounds, val_only)
            gains[f"{GAIN_OF_TEN}, {LOSS_OF_TEN}"] = response_df
        if gain == f"{LOSS_OF_THIRTY}, {GAIN_UNDER_TEN}":
            gain_under_and_loss_ten_trials = aoi_analysis_df.loc[get_loss_of_thirty(aoi_analysis_df) & get_gain_of_less_than_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_under_and_loss_ten_trials, with_error_bounds, val_only)
            gains[f"{LOSS_OF_THIRTY}, {GAIN_UNDER_TEN}"] = response_df
        if gain == f"{LOSS_OF_THIRTY}, {GAIN_OF_TEN}":
            gain_under_and_loss_ten_trials = aoi_analysis_df.loc[get_loss_of_thirty(aoi_analysis_df) & get_gain_of_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_under_and_loss_ten_trials, with_error_bounds, val_only)
            gains[f"{LOSS_OF_THIRTY}, {GAIN_OF_TEN}"] = response_df
    return DataFrame.from_dict(gains, orient="index")
