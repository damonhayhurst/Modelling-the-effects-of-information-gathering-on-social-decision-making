

import itertools
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, concat
from scipy.stats import ttest_ind, false_discovery_control
from utils.display import display
from utils.masks import get_gain_of_less_than_ten, get_gain_of_thirty, get_gain_of_twenty, get_gain_under_thirty, get_loss_of_less_than_ten, get_loss_of_ten, get_loss_of_thirty, get_loss_under_thirty, get_negative_gain, get_positive_gain, get_positive_gain_of_less_than_ten
from utils.columns import CLUSTER, CLUSTER_1, CLUSTER_2, DOF, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_OF_TWENTY, GAIN_UNDER_TEN, GAIN_UNDER_THIRTY, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_OF_TWENTY, LOSS_UNDER_TEN, LOSS_UNDER_THIRTY, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, NEGATIVE_GAIN, OTHER_LIE, OTHER_TRUTH, P_VALUE, POSITIVE_GAIN, RT, SELECTED_AOI, SELF_LIE, SELF_TRUE, T_STATISTIC, TRIAL_COUNT
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


def is_equal_var(stat: str, levenes: dict[str, bool]):
    if not levenes:
        return True
    elif stat not in levenes:
        return True
    else:
        return levenes[stat].pvalue > 0.05


def t_test(stat: str, a: DataFrame, b: DataFrame, levenes: dict[str, bool]):
    equal_var = is_equal_var(stat, levenes)
    t_test = ttest_ind(a[stat], b[stat], equal_var=equal_var)
    return {
        T_STATISTIC: t_test.statistic,
        P_VALUE: t_test.pvalue,
        DOF: t_test.df
    }


def get_t_test_response_stats(a: DataFrame, b: DataFrame, levenes: dict[str, bool] = None):
    return DataFrame({
        SELF_LIE: t_test(SELF_LIE, a, b, levenes),
        SELF_TRUE: t_test(SELF_TRUE, a, b, levenes),
        OTHER_LIE: t_test(OTHER_LIE, a, b, levenes),
        OTHER_TRUTH: t_test(OTHER_TRUTH, a, b, levenes),
        N_TRANSITIONS: t_test(N_TRANSITIONS, a, b, levenes),
    })


def get_gains_response_stats(aoi_analysis_df, gain_labels: list[str], with_error_bounds: bool = True, val_only: bool = False):
    gains = {}
    for gain in gain_labels:
        gain_df = get_trials_by_condition(gain, aoi_analysis_df)
        response_df = get_response_stats(gain_df, with_error_bounds, val_only)
        gains[gain] = response_df
        
    return DataFrame.from_dict(gains, orient="index")


def get_trials_by_condition(label: str, aoi_analysis_df: DataFrame):
    conditions = {
        GAIN_OF_TEN: aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df)],
        GAIN_UNDER_TEN: aoi_analysis_df.loc[get_gain_of_less_than_ten(aoi_analysis_df)],
        GAIN_OF_TWENTY: aoi_analysis_df.loc[get_gain_of_twenty(aoi_analysis_df)],
        GAIN_OF_THIRTY: aoi_analysis_df.loc[get_gain_of_thirty(aoi_analysis_df)],
        GAIN_UNDER_THIRTY: aoi_analysis_df.loc[get_gain_under_thirty(aoi_analysis_df)],
        NEGATIVE_GAIN: aoi_analysis_df.loc[get_negative_gain(aoi_analysis_df)],
        POSITIVE_GAIN: aoi_analysis_df.loc[get_positive_gain(aoi_analysis_df)],
        LOSS_OF_TEN:aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df)],
        LOSS_UNDER_TEN:aoi_analysis_df.loc[get_loss_of_less_than_ten(aoi_analysis_df)],
        LOSS_OF_TWENTY:aoi_analysis_df.loc[get_gain_of_twenty(aoi_analysis_df)],
        LOSS_OF_THIRTY:aoi_analysis_df.loc[get_loss_of_thirty(aoi_analysis_df)],
        LOSS_UNDER_THIRTY:aoi_analysis_df.loc[get_loss_under_thirty(aoi_analysis_df)],
        f"{GAIN_UNDER_TEN}, {LOSS_UNDER_TEN}":aoi_analysis_df.loc[get_gain_of_less_than_ten(aoi_analysis_df) & get_loss_of_less_than_ten(aoi_analysis_df)],
        f"{GAIN_UNDER_TEN}, {LOSS_OF_TEN}":aoi_analysis_df.loc[get_gain_of_less_than_ten(aoi_analysis_df) & get_loss_of_ten(aoi_analysis_df)],
        f"{GAIN_OF_TEN}, {LOSS_UNDER_TEN}":aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df) & get_loss_of_less_than_ten(aoi_analysis_df)],
        f"{GAIN_OF_TEN}, {LOSS_OF_TEN}":aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df) & get_loss_of_ten(aoi_analysis_df)],
        f"{LOSS_OF_THIRTY}, {GAIN_UNDER_TEN}":aoi_analysis_df.loc[get_loss_of_thirty(aoi_analysis_df) & get_gain_of_less_than_ten(aoi_analysis_df)],
        f"{LOSS_OF_THIRTY}, {GAIN_OF_TEN}":aoi_analysis_df.loc[get_loss_of_thirty(aoi_analysis_df) & get_gain_of_ten(aoi_analysis_df)]
    }
    return conditions[label]