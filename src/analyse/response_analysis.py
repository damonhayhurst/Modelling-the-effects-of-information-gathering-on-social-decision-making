
import itertools
from typing import Callable
import matplotlib.pyplot as plt
import os
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import XKCD_COLORS
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
from pandas import DataFrame, Index, MultiIndex, Series, concat, get_dummies
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from analyse.get_response_stats import get_gains_response_stats, get_is_lie, get_response_stats_for_clusters_by_pid, get_response_stats_for_clusters_by_trial_id, get_t_test_response_stats, get_response_stats, get_response_stats_for_clusters, get_response_stats_for_group_by, get_trials_by_condition, t_test
from utils.columns import AOI, CHI_SQUARED, CLUSTER, CLUSTER_1, CLUSTER_2, CONDITION, CONDITION_1, CONDITION_2, DISTANCE, DOF, DWELL_TIME, FREQUENCIES, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_OF_TWENTY, GAIN_UNDER_TEN, GAIN_UNDER_THIRTY, GROUP, IS_LIE, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_OF_TWENTY, LOSS_UNDER_TEN, LOSS_UNDER_THIRTY, N, N_TRANSITIONS, NEGATIVE_GAIN, OTHER_LIE, OTHER_LOSS, OTHER_TRUTH, P_VALUE, PID, PID_1, PID_2, POSITIVE_GAIN, RT, SELECTED_AOI, SELF_GAIN, SELF_LIE, SELF_TRUE, T_STATISTIC, TRIAL_COUNT, TRIAL_ID, TRUTH
from utils.display import display
from utils.masks import get_gain_of_between_ten_and_thirty, get_gain_of_between_ten_and_twenty, get_gain_of_between_twenty_and_thirty, get_gain_of_thirty, get_gain_under_thirty, get_loss_of_between_ten_and_thirty, get_loss_of_between_twenty_and_thirty, get_loss_of_twenty, get_loss_under_thirty, get_positive_gain
from utils.masks import get_gain_of_ten
from utils.masks import get_gain_of_twenty
from utils.masks import get_positive_gain_of_less_than_ten
from utils.masks import get_gain_of_less_than_ten
from utils.masks import get_negative_gain
from utils.masks import get_loss_of_ten
from utils.masks import get_loss_of_less_than_ten
from utils.masks import get_loss_of_thirty
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import levene, false_discovery_control


XKCD_COLORS_LIST = list(XKCD_COLORS.values())


def calculate_mean_response_stat(df: DataFrame, stat: str, with_error_bounds: bool = True):
    mean = df[stat].apply(lambda x: x[0]).mean()

    if with_error_bounds:
        mean_lower_error = df.apply(lambda x: x[1][0]).mean()
        mean_upper_error = df.apply(lambda x: x[1][1]).mean()
        return mean, (mean_lower_error, mean_upper_error)
    else:
        agg_std = get_aggregate_std(df, stat)
        return mean, agg_std


def get_aggregate_std(df, stat: str):
    means = df[stat].apply(lambda x: x[0])
    stds = df[stat].apply(lambda x: x[1])
    counts = df[TRIAL_COUNT]

    # Overall mean calculation
    total_samples = np.sum(counts)
    overall_mean = np.sum(means * counts) / total_samples

    # Within-group variance contribution
    within_variance = np.sum((counts - 1) * stds**2)

    # Between-group variance contribution
    between_variance = np.sum(counts * (means - overall_mean)**2)

    # Aggregate standard deviation calculation
    aggregate_variance = (within_variance + between_variance) / (total_samples - 1)
    aggregate_std_dev = np.sqrt(aggregate_variance)

    return aggregate_std_dev


def get_confidence_intervals(df: DataFrame):
    val = df.apply(lambda x: x[0])
    confidence_level = 0.95
    n = len(df)
    mean = np.mean(val)
    std_err = stats.sem(val)
    margin_of_error = std_err * stats.t.ppf((1 + confidence_level) / 2., n-1)
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)
    return confidence_interval


def get_trial_response_stats_by_pid(cluster_df: DataFrame, analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    cluster_df.index = analysis_df.index
    response_df = get_response_stats_for_clusters_by_pid(cluster_df.groupby([PID, CLUSTER]), analysis_df, analysis_df.index, with_error_bounds, val_only)
    return response_df


def get_response_stats_by_trial_id(cluster_df: DataFrame, analysis_df: DataFrame):
    by_trial_id_df = cluster_df.groupby(level=TRIAL_ID)
    response_df = get_response_stats_for_clusters_by_trial_id(cluster_df.groupby([TRIAL_ID, CLUSTER]), analysis_df, analysis_df.index)
    response_df[CLUSTER] = cluster_df[CLUSTER]
    return response_df.sort_values(CLUSTER)


def get_response_stats_by_pid(cluster_df: DataFrame, analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    by_pid_df = analysis_df.groupby(PID)
    response_df = get_response_stats_for_group_by(by_pid_df, with_error_bounds, val_only)
    response_df[CLUSTER] = cluster_df[CLUSTER]
    return response_df.sort_values(CLUSTER)


def get_trial_response_stats_no_clusters(analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = True):
    return get_response_stats(analysis_df, with_error_bounds, val_only)


def get_pid_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index.get_level_values(PID), with_error_bounds, val_only)


def get_trial_id_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index.get_level_values(TRIAL_ID))


def get_trial_count_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    analysis_df = analysis_df.reset_index().set_index([PID, TRIAL_COUNT])
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index.get_level_values(TRIAL_COUNT))


def get_trial_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    cluster_df.index = analysis_df.index
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index, with_error_bounds, val_only)


def get_trial_response_stats_for_clusters_by_gain_label(cluster_df: DataFrame, gain_label: str, analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    filtered_analysis_df = get_trials_by_condition(gain_label, analysis_df)
    cluster_df = cluster_df.loc[filtered_analysis_df.index]
    return get_trial_response_stats_for_clusters(cluster_df, filtered_analysis_df, with_error_bounds, val_only)


def get_trial_id_response_stats_no_clusters(analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    return get_response_stats_for_group_by(analysis_df.groupby(TRIAL_ID), with_error_bounds, val_only)


def get_pid_response_stats_for_clusters_by_gain_label(cluster_df: DataFrame, gain_label: str, analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    if gain_label == GAIN_UNDER_TEN:
        filtered_analysis_df = analysis_df.loc[get_gain_of_less_than_ten(analysis_df)]
    if gain_label == GAIN_OF_TEN:
        filtered_analysis_df = analysis_df.loc[get_gain_of_ten(analysis_df)]
    return get_pid_response_stats_for_clusters(cluster_df, filtered_analysis_df, with_error_bounds, val_only)


def get_pid_response_stats_no_clusters(analysis_df: DataFrame, with_error_bounds: bool = True, val_only: bool = False):
    return get_response_stats_for_group_by(analysis_df.groupby(PID), with_error_bounds, val_only)


def get_trials_by_cluster(cluster: int, cluster_df: DataFrame, analysis_df: DataFrame):
    cluster_df.index = analysis_df.index
    by_cluster_df = cluster_df.groupby(CLUSTER)
    idx = by_cluster_df.get_group(cluster).index
    return analysis_df.loc[analysis_df.index.isin(idx)]


def sort_response_df_by_pid_lie_percent(response_df: DataFrame, analysis_df: DataFrame):
    lie_idx = get_response_stats_for_group_by(analysis_df.groupby(PID)).sort_values(LIE).index
    return response_df.reindex(index=lie_idx, level=PID)


def sort_response_df_by_pid_lie_percent_in_cluster(response_df: DataFrame, analysis_df: DataFrame):
    no_cluster_df = get_response_stats_for_group_by(analysis_df.groupby(PID))
    no_cluster_df[CLUSTER] = response_df[CLUSTER]
    lie_idx = no_cluster_df.sort_values([CLUSTER, LIE]).index
    return response_df.reindex(index=lie_idx, level=PID)


def plot_percent_lies_by_trial_id(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(15, 6))
    # plt.title('%sPercent of Lies by TRIAL ID' % (title_prefix))
    plt.ylabel('Percent')
    plt.xlabel('Trial ID')
    plt.ylim(0, 100)
    plot_response_stats(ax, responses_df, [LIE], colors=colors, to_file=to_file)


def plot_percent_lies_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    # plt.title('%sPercent of Lies by Participant ID' % (title_prefix))
    plt.ylabel('Percent')
    plt.xlabel('Participant ID')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [LIE], colors, to_file)


def plot_n_transitions_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    # plt.title('%sPercent of Lies by Participant ID' % (title_prefix))
    plt.ylabel('Count')
    plt.xlabel('Participant ID')
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [N_TRANSITIONS], colors, to_file)


def plot_aoi_by_pid(responses_df: DataFrame,  aoi: str, colors: list[str] = XKCD_COLORS_LIST, dwell_as_a_proportion: bool = False, to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%s' % (aoi))
    plt.ylabel('Dwell Time / Reaction Time') if dwell_as_a_proportion else plt.ylabel('Dwell Time (ms)')
    plt.xlabel('Participant ID')
    plt.ylim(0, 1300) if not dwell_as_a_proportion else plt.ylim()
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [aoi], colors, to_file)


def plot_percent_lies_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('%sPercent of Lies per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [LIE], colors, to_file)


def plot_gain_of_ten_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    # plt.title('%sPercent Gain < 10 by PID' % (title_prefix))
    plt.ylabel('N Trials')
    plt.ylim(0, 80)
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [GAIN_OF_TEN], colors, to_file)


def plot_gain_under_ten_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    # plt.title('%sPercent Gain + 10 by PID' % (title_prefix))
    plt.ylabel('N Trials')
    plt.ylim(0, 80)
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [GAIN_UNDER_TEN], colors, to_file)


def plot_gain_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%sPercent Gain per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [GAIN_OF_TEN, GAIN_UNDER_TEN], colors, to_file)


def plot_percent_lie_by_gain_for_clusters(gain_under_ten_responses_df: DataFrame, gain_of_ten_responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    gain_under_ten_responses_df[GAIN_OF_TEN] = gain_of_ten_responses_df[LIE]
    gain_under_ten_responses_df[GAIN_UNDER_TEN] = gain_under_ten_responses_df[LIE]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, gain_under_ten_responses_df, [GAIN_OF_TEN, GAIN_UNDER_TEN], colors, to_file)


def plot_n_transitions_by_gain_for_clusters(gain_under_ten_responses_df: DataFrame, gain_of_ten_responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    gain_under_ten_responses_df[GAIN_OF_TEN] = gain_of_ten_responses_df[N_TRANSITIONS]
    gain_under_ten_responses_df[GAIN_UNDER_TEN] = gain_under_ten_responses_df[N_TRANSITIONS]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel('Count')
    plot_response_stats_for_clusters(ax, gain_under_ten_responses_df, [GAIN_OF_TEN, GAIN_UNDER_TEN], colors, to_file)


def plot_dwell_time_by_gain_for_clusters(gain_under_ten_responses_df: DataFrame, gain_of_ten_responses_df: DataFrame, aoi: str, dwell_as_a_proportion: bool = False, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    gain_under_ten_responses_df[GAIN_OF_TEN] = gain_of_ten_responses_df[aoi]
    gain_under_ten_responses_df[GAIN_UNDER_TEN] = gain_under_ten_responses_df[aoi]
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.title('%s' % aoi)
    plt.ylabel('Dwell Time / Reaction Time') if dwell_as_a_proportion else plt.ylabel('Dwell Time (ms)')
    plt.ylim(0, 1200) if not dwell_as_a_proportion else plt.ylim()
    plt.yticks([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]) if not dwell_as_a_proportion else plt.yticks()
    plot_response_stats_for_clusters(ax, gain_under_ten_responses_df, [GAIN_OF_TEN, GAIN_UNDER_TEN], colors, to_file)


def plot_n_trials_by_gain_for_clusters(gain_under_ten_responses_df: DataFrame, gain_of_ten_responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    gain_under_ten_responses_df[GAIN_OF_TEN] = gain_of_ten_responses_df[TRIAL_COUNT]
    gain_under_ten_responses_df[GAIN_UNDER_TEN] = gain_under_ten_responses_df[TRIAL_COUNT]
    fig, ax = plt.subplots(figsize=(12, 9))
    plt.ylabel('N Trials')
    plot_response_stats_for_clusters(ax, gain_under_ten_responses_df, [GAIN_OF_TEN, GAIN_UNDER_TEN], colors, to_file)


def plot_dwell_times_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, dwell_as_a_proportion: bool = False, to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('%sAverage Dwell Time for each AOI per %s Cluster' % (title_prefix, cluster_by))
    plt.ylim(0, 1400) if not dwell_as_a_proportion else plt.ylim()
    plt.ylabel('Dwell Time / Reaction Time') if dwell_as_a_proportion else plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_clusters(ax, responses_df, [SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH], colors, to_file)


def plot_n_transitions_for_clusters(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title('%sNumber of Transitions per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Count')
    plot_response_stats_for_clusters(ax, response_df, [N_TRANSITIONS], colors, to_file)


def plot_n_trials_for_clusters(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', sort_by: str = None, to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.ylabel('N Trials')
    response_df = response_df.sort_values(sort_by) if sort_by else response_df
    plot_response_stats_for_clusters(ax, response_df, [TRIAL_COUNT], colors, to_file)


def plot_n_trials_for_clusters_by_pid(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', title_override: str = None, to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    # plt.title('%sQuantity per %s Cluster' % (title_prefix, cluster_by))
    if title_override:
        plt.title(title_override)
    plt.ylabel('N Trials')
    plt.xlabel('PID')
    plot_response_stats_for_clusters_by_pid(ax, response_df, [TRIAL_COUNT], colors, to_file)


def plot_n_trials_for_clusters_by_pid_sort_by_lie(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    plot_n_trials_for_clusters_by_pid(response_df, cluster_by, colors=colors, title_prefix=title_prefix, sort_by=LIE, to_file=to_file)


def create_error_bars_y(values, errors):
    error_bars = []
    for v in range(len(values)):
        value = values[v]
        lower, upper = errors[v][0], errors[v][1]
        error_bars.append([value - lower, upper - value])
    return np.array(error_bars).T


def plot_response_stats(ax: Axes, responses_df: DataFrame, stats: list[str], group_col: str = None, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    num_categories = len(stats)
    bar_width = 0.3
    indices = np.arange(num_categories)
    enumerable_df = responses_df.reset_index()
    is_grouped = GROUP in enumerable_df.columns

    if group_col:
        group_vals = enumerable_df[group_col].unique()
        groups = {val: idx for idx, val in enumerate(group_vals)}

    for i, value in enumerate(enumerable_df.index):
        values = [enumerable_df.loc[value][stat][0] for stat in stats]
        if group_col == GROUP:
            group = enumerable_df.loc[value][GROUP]
            ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[groups[group]], label=f'{group}', ecolor="black")
        elif group_col == CLUSTER:
            cluster = int(enumerable_df.loc[value][CLUSTER])
            ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[cluster - 1], label=f'Cluster {cluster}', ecolor="black")
        else:
            ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[0])
        errors = [enumerable_df.loc[value][stat][1] for stat in stats]
        errors_y = create_error_bars_y(values, errors)
        ax.errorbar(indices + i * bar_width, values, errors_y, ecolor="black", capsize=5)

    plt.xticks([])
    if len(stats) > 1:
        plt.xticks(((bar_width/2 * len(responses_df.index)) + indices) - bar_width/2, stats, rotation=0)
    else:
        ax.set_xticks((bar_width/2) * (np.array(range(0, len(responses_df.index))) * 2))
        ax.set_xticklabels(responses_df.index, fontsize=8)

    if group_col == GROUP:
        plt.legend(title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), title='Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    elif group_col == CLUSTER:
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    # ax.spines['left'].set_linewidth(1.2)
    # ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2)

    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)

    plt.show()


def plot_response_stats_for_groups(ax: Axes, responses_df: DataFrame, stats: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    plot_response_stats(ax, responses_df, stats, GROUP, colors, to_file)


def plot_response_stats_for_clusters(ax: Axes, responses_df: DataFrame, stats: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    plot_response_stats(ax, responses_df, stats, CLUSTER, colors, to_file)


def plot_response_stats_for_clusters_by_pid(ax: Axes, responses_df: DataFrame, stats: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    bar_width = 0.3
    gap_width = 0.4  # Width of the gap between different PIDs
    responses_df = responses_df.reset_index()
    num_categories = len(stats)
    is_clustered = CLUSTER in responses_df.columns
    pids = responses_df[PID].unique()
    current_position = 0

    for pid in pids:
        pid_df = responses_df[responses_df[PID] == pid]
        indices = np.arange(num_categories)

        for i, value in enumerate(pid_df.index):
            values = [pid_df.loc[value][stat][0] for stat in stats]
            cluster = int(pid_df.loc[value][CLUSTER]) if is_clustered else 0
            ax.bar(indices + current_position, values, width=bar_width, color=colors[cluster - 1], label=f'Cluster {cluster}')
            errors = [pid_df.loc[value][stat][1] for stat in stats]
            errors_y = create_error_bars_y(values, errors)
            ax.errorbar(indices + current_position, values, errors_y, ecolor='black', capsize=5)
            current_position += bar_width

        current_position += gap_width  # Add a gap after each PID

    # plt.xticks([])
    # if len(stats) > 1:
        # plt.xticks(((bar_width/2 * len(responses_df.index)) + np.arange(len(pids) * (num_categories + 1))) - bar_width/2, stats, rotation=0)

    if is_clustered:
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Adding PID labels
    current_position = 0
    pid_positions = []
    for pid in pids:
        pid_df = responses_df[responses_df[PID] == pid]
        pid_positions.append(current_position + (len(pid_df) / 2 * bar_width) / 2)
        current_position += (len(pid_df) * bar_width) + gap_width

    ax.set_xticks(pid_positions)
    ax.set_xticklabels(pids, fontsize=15)

    ax.tick_params(axis='both', which='both', length=0)

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_linewidth(1.2)
    # ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2)

    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)

    plt.show()


def plot_avg_dwell_time_distributions(aoi_analysis_df: DataFrame, to_file: str = None):
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    # fig.suptitle("Frequency of Average Dwell Time (ms) for each AOI per Trial < 1000ms")
    all_data = aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]]
    min_x, max_x = np.min(all_data.values), np.max(all_data.values)

    max_y = 0
    axes = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
    titles = [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]
    for ax, title in zip(axes, titles):
        col = aoi_analysis_df[title]
        sns.histplot(col[col != 0], kde=True, ax=ax, bins=300, color=XKCD_COLORS["xkcd:light blue"], edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("")

        # Update maximum y-value for uniform scaling
        current_max_y = ax.get_ylim()[1]
        max_y = max(max_y, current_max_y)

    for ax in axs.flat:
        ax.set_xlim(200, 1000)
        ax.set_ylim(0, max_y)

    # Save the plot to a file if a path is provided
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)

    plt.tight_layout()
    plt.show()


def plot_n_transitions_distributions(aoi_analysis_df: DataFrame, to_file: str = None):
    plt.figure(figsize=(10, 6))
    # plt.title("Frequency of Number of Transitions per Trial")
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    sns.barplot(aoi_analysis_df[N_TRANSITIONS].value_counts(), capsize=5, color=XKCD_COLORS["xkcd:light blue"], edgecolor="black")
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def plot_rt_distributions(aoi_analysis_df: DataFrame, to_file: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(aoi_analysis_df[RT], kde=True, color=XKCD_COLORS["xkcd:light blue"], bins=200, edgecolor="black")
    # plt.title('Distribution of Reaction Times')
    plt.xlabel('Duration (ms)')
    plt.ylabel('Frequency')
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def plot_distance_distributions(distance_df: DataFrame, to_file: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(distance_df[DISTANCE], kde=True)
    plt.title('Distribution of Distance')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def plot_dwell_time_distribution(dwell_df: DataFrame, to_file: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(dwell_df[dwell_df[AOI] != None][DWELL_TIME], kde=True, bins=1000)
    plt.title('Distribution of Dwell Time')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.xlim(0, 0.1)
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def plot_gains_mean_percent_lie(aoi_analysis_df, gain_labels: list[str], title: str, colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains = get_gains_response_stats(aoi_analysis_df, gain_labels)
    gains_to_show = gains[gains.index.isin(gain_labels)]
    display(gain_labels)
    display(gains)
    simple_plot(gains_to_show[LIE], gain_labels, title=title, ylabel="Percent Lie", colors=colors, to_file=to_file)


def plot_gains_avg_dwell_time(analysis_df: DataFrame, gain_labels: list[str], is_dwell_a_proportion: bool = False, colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([GAIN_UNDER_TEN, GAIN_OF_TEN, POSITIVE_GAIN, NEGATIVE_GAIN], name=GROUP), columns=[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH])
    gain_of_ten_trials = analysis_df.loc[get_gain_of_ten(analysis_df)]
    gain_under_ten_trials = analysis_df.loc[get_gain_of_less_than_ten(analysis_df)]
    positive_gain_trials = analysis_df.loc[get_positive_gain(analysis_df)]
    negative_gain_trials = analysis_df.loc[get_negative_gain(analysis_df)]
    gain_of_thirty = analysis_df.loc[get_gain_of_thirty(analysis_df)]
    gain_under_thirty = analysis_df.loc[get_gain_under_thirty(analysis_df)]
    gains_df.loc[GAIN_OF_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(gain_of_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[GAIN_UNDER_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(gain_under_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[POSITIVE_GAIN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(positive_gain_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[NEGATIVE_GAIN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(negative_gain_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[GAIN_OF_THIRTY, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(gain_of_thirty)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[GAIN_UNDER_THIRTY, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(gain_under_thirty)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]

    fig, ax = plt.subplots(figsize=(28, 6))
    # plt.title('Average Dwell Time for each AOI per differing quantities of net gain to sender')
    plt.ylabel('Dwell Time / Reaction Time') if is_dwell_a_proportion else plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_groups(ax, gains_df.loc[gain_labels], [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH], colors, to_file)


def plot_gains_n_transitions(analysis_df: DataFrame, gain_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([GAIN_UNDER_TEN, GAIN_OF_TEN, POSITIVE_GAIN, NEGATIVE_GAIN], name=GROUP), columns=[N_TRANSITIONS])
    gain_of_ten_trials = analysis_df.loc[get_gain_of_ten(analysis_df)]
    gain_under_ten_trials = analysis_df.loc[get_gain_of_less_than_ten(analysis_df)]
    positive_gain_trials = analysis_df.loc[get_positive_gain(analysis_df)]
    negative_gain_trials = analysis_df.loc[get_negative_gain(analysis_df)]
    gain_of_thirty = analysis_df.loc[get_gain_of_thirty(analysis_df)]
    gain_under_thirty = analysis_df.loc[get_gain_under_thirty(analysis_df)]
    gains_df = DataFrame(index=Index([GAIN_UNDER_TEN, GAIN_OF_TEN], name=GROUP), columns=[N_TRANSITIONS])
    gains_df.loc[GAIN_OF_TEN, N_TRANSITIONS] = get_response_stats(gain_of_ten_trials)[N_TRANSITIONS]
    gains_df.loc[GAIN_UNDER_TEN, N_TRANSITIONS] = get_response_stats(gain_under_ten_trials)[N_TRANSITIONS]
    gains_df.loc[POSITIVE_GAIN, N_TRANSITIONS] = get_response_stats(positive_gain_trials)[N_TRANSITIONS]
    gains_df.loc[NEGATIVE_GAIN, N_TRANSITIONS] = get_response_stats(negative_gain_trials)[N_TRANSITIONS]
    gains_df.loc[GAIN_OF_THIRTY, N_TRANSITIONS] = get_response_stats(gain_of_thirty)[N_TRANSITIONS]
    gains_df.loc[GAIN_UNDER_THIRTY, N_TRANSITIONS] = get_response_stats(gain_under_thirty)[N_TRANSITIONS]

    fig, ax = plt.subplots(figsize=(25, 6))
    # plt.title('Average N Transitions per differing quantities of net gain to sender')
    plt.ylabel('N Transitions')
    plot_response_stats_for_groups(ax, gains_df.loc[gain_labels], [N_TRANSITIONS], colors, to_file)


def plot_losses_avg_dwell_time(analysis_df: DataFrame, loss_labels: list[str], is_dwell_a_proportion: bool = False, colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([LOSS_UNDER_TEN, LOSS_OF_TEN, LOSS_OF_TWENTY, LOSS_OF_THIRTY, LOSS_UNDER_THIRTY], name=GROUP), columns=[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH])
    loss_of_ten_trials = analysis_df.loc[get_loss_of_ten(analysis_df)]
    loss_under_ten_trials = analysis_df.loc[get_loss_of_less_than_ten(analysis_df)]
    loss_of_twenty_trials = analysis_df.loc[get_loss_of_twenty(analysis_df)]
    loss_of_thirty_trials = analysis_df.loc[get_loss_of_thirty(analysis_df)]
    loss_under_thirty_trials = analysis_df.loc[get_loss_under_thirty(analysis_df)]
    gains_df.loc[LOSS_OF_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(loss_of_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[LOSS_UNDER_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(loss_under_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[LOSS_OF_TWENTY, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(loss_of_twenty_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[LOSS_OF_THIRTY, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(loss_of_thirty_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[LOSS_UNDER_THIRTY, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(loss_under_thirty_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]

    fig, ax = plt.subplots(figsize=(28, 6))
    # plt.title('Average Dwell Time for each AOI per differing quantities of net loss to reciever')
    plt.ylabel('Dwell Time / Reaction Time') if is_dwell_a_proportion else plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_groups(ax, gains_df.loc[loss_labels], [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH], colors, to_file)


def plot_losses_n_transitions(analysis_df: DataFrame, loss_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([LOSS_UNDER_TEN, LOSS_OF_TEN, LOSS_OF_TWENTY, LOSS_OF_THIRTY], name=GROUP), columns=[N_TRANSITIONS])
    loss_of_ten_trials = analysis_df.loc[get_loss_of_ten(analysis_df)]
    loss_under_ten_trials = analysis_df.loc[get_loss_of_less_than_ten(analysis_df)]
    loss_of_twenty_trials = analysis_df.loc[get_loss_of_twenty(analysis_df)]
    loss_of_thirty_trials = analysis_df.loc[get_loss_of_thirty(analysis_df)]
    loss_under_thirty_trials = analysis_df.loc[get_loss_under_thirty(analysis_df)]
    gains_df.loc[LOSS_OF_TEN, N_TRANSITIONS] = get_response_stats(loss_of_ten_trials)[N_TRANSITIONS]
    gains_df.loc[LOSS_UNDER_TEN, N_TRANSITIONS] = get_response_stats(loss_under_ten_trials)[N_TRANSITIONS]
    gains_df.loc[LOSS_OF_TWENTY, N_TRANSITIONS] = get_response_stats(loss_of_twenty_trials)[N_TRANSITIONS]
    gains_df.loc[LOSS_OF_THIRTY, N_TRANSITIONS] = get_response_stats(loss_of_thirty_trials)[N_TRANSITIONS]
    gains_df.loc[LOSS_UNDER_THIRTY, N_TRANSITIONS] = get_response_stats(loss_under_thirty_trials)[N_TRANSITIONS]

    fig, ax = plt.subplots(figsize=(25, 6))
    # plt.title('Average N Transitions per differing quantities of net loss to reciever')
    plt.ylabel('N Transitions')
    plot_response_stats_for_groups(ax, gains_df.loc[loss_labels], [N_TRANSITIONS], colors, to_file)


def simple_plot(x, xlabel, title, ylabel, yticks=[], colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    vals = [val[0] for val in x]
    errors = [val[1] for val in x]
    yerr = create_error_bars_y(vals, errors)
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.bar(xlabel, vals, yerr=yerr, capsize=5, color=colors)
    plt.xticks(xlabel, fontsize=15)
    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def do_levenes_test(trials1, trials2):
    levenes = {}
    levenes[SELF_LIE] = levene(trials1[SELF_LIE], trials2[SELF_LIE])
    levenes[SELF_TRUE] = levene(trials1[SELF_TRUE], trials2[SELF_TRUE])
    levenes[OTHER_LIE] = levene(trials1[OTHER_LIE], trials2[OTHER_LIE])
    levenes[OTHER_TRUTH] = levene(trials1[OTHER_TRUTH], trials2[OTHER_TRUTH])
    levenes[N_TRANSITIONS] = levene(trials1[N_TRANSITIONS], trials2[N_TRANSITIONS])
    return levenes


def do_gains_t_test(analysis_df: DataFrame, for_test: list[str]):
    t_tests_df = do_combination_t_test(analysis_df, for_test, get_trials_by_condition)
    t_tests_df.index = MultiIndex.from_tuples(t_tests_df.index.values, names=[CONDITION_1, CONDITION_2])
    return t_tests_df


def do_pids_t_test(analysis_df: DataFrame):
    group_by_pid = analysis_df.groupby(PID)
    pids = group_by_pid.groups.keys()
    pid_combs = list(itertools.combinations(pids, 2))
    def get_pid_trials(pid, analysis_df): return analysis_df.xs(pid, level=PID)

    t_tests_df = do_combination_t_test(analysis_df, pid_combs, get_pid_trials)
    t_tests_df.index = MultiIndex.from_tuples(t_tests_df.index.values, names=[PID_1, PID_2])
    return t_tests_df


def do_combination_t_test(analysis_df: DataFrame, for_test_labels: list[str], get_trials_fn: Callable[[str, DataFrame], DataFrame]):
    t_tests_df = DataFrame()
    for label1, label2 in for_test_labels:
        trials1, trials2 = get_trials_fn(label1, analysis_df), get_trials_fn(label2, analysis_df)
        t_test_df = do_t_test_for_response_stats(trials1, trials2)
        t_test_df.index = [(label1, label2)]
        t_tests_df = concat([t_tests_df, t_test_df])

    t_tests_df = correct_p_values(t_tests_df)
    return t_tests_df


def do_t_test_for_response_stats(trials_df: DataFrame, other_trials_df: DataFrame):
    levenes = do_levenes_test(trials_df, other_trials_df)
    # display(levenes)
    t_test_df = get_t_test_response_stats(trials_df, other_trials_df, levenes=levenes)

    result = concat({T_STATISTIC: t_test_df.loc[T_STATISTIC], P_VALUE: t_test_df.loc[P_VALUE], DOF: t_test_df.loc[DOF]}, axis=1)
    return result.T.stack().to_frame().T


def correct_p_values(df: DataFrame):
    df[P_VALUE] = false_discovery_control(df[P_VALUE])
    return df


def get_aggregate_cluster_t_test_response_stats(t_test_df: DataFrame):
    cluster_values = set(t_test_df[CLUSTER_1].tolist() + t_test_df[CLUSTER_2].tolist())
    combs = list(itertools.combinations_with_replacement(cluster_values, 2))
    agg_t_test_df = DataFrame()
    for cluster1, cluster2 in combs:
        is_cluster_comb = ((t_test_df[CLUSTER_1] == cluster1) & (t_test_df[CLUSTER_2] == cluster2)) | ((t_test_df[CLUSTER_1] == cluster2) & (t_test_df[CLUSTER_2] == cluster1))
        cluster_t_tests_df = t_test_df.loc[is_cluster_comb]
        cluster_t_test_df = DataFrame.from_dict({T_STATISTIC: cluster_t_tests_df[T_STATISTIC].mean(), P_VALUE: cluster_t_tests_df[P_VALUE].mean(), DOF: len(cluster_t_tests_df)})
        cluster_t_test_df = cluster_t_test_df.T.stack().to_frame().T
        cluster_t_test_df.index = [(cluster1, cluster2)]
        agg_t_test_df = concat([agg_t_test_df, cluster_t_test_df])

    # agg_t_test_df = correct_p_values(agg_t_test_df)
    agg_t_test_df.index = MultiIndex.from_tuples(agg_t_test_df.index.values, names=[CLUSTER_1, CLUSTER_2])
    return agg_t_test_df


def do_clustered_pid_t_test(analysis_df: DataFrame, cluster_df: DataFrame):
    t_tests_df = do_pids_t_test(analysis_df)
    t_tests_df[CLUSTER_1] = cluster_df.loc[t_tests_df.index.get_level_values(PID_1), CLUSTER].values
    t_tests_df[CLUSTER_2] = cluster_df.loc[t_tests_df.index.get_level_values(PID_2), CLUSTER].values
    return get_aggregate_cluster_t_test_response_stats(t_tests_df)


def do_clustered_by_pid_mean_t_test(analysis_df: DataFrame, cluster_df: DataFrame, stats: list[str]):
    group_by_pid = analysis_df.groupby(PID)
    response_stats_by_pid = get_response_stats_for_group_by(group_by_pid, with_error_bounds=False, val_only=True)
    response_stats_by_pid[CLUSTER] = cluster_df.loc[response_stats_by_pid.index]
    group_by_cluster = response_stats_by_pid.groupby(CLUSTER)
    clusters = group_by_cluster.groups.keys()
    cluster_combs = list(itertools.combinations(clusters, 2))
    t_tests_df = DataFrame()
    for cluster1, cluster2 in cluster_combs:
        cluster_trials1, cluster_trials2 = group_by_cluster.get_group(cluster1), group_by_cluster.get_group(cluster2)
        levenes = do_levenes_test(cluster_trials1, cluster_trials2)
        t_test_df = DataFrame({stat: t_test(stat, cluster_trials1, cluster_trials2, levenes) for stat in stats})
        t_test_df = t_test_df.stack().to_frame().T
        t_test_df.index = [(cluster1, cluster2)]
        t_tests_df = concat([t_tests_df, t_test_df])

    t_tests_df = correct_p_values(t_tests_df)
    t_tests_df.index = MultiIndex.from_tuples(t_tests_df.index.values, names=[CLUSTER_1, CLUSTER_2])
    return t_tests_df


def print_chi_squared_stats(test_df):
    print(test_df.name)
    print(f"Chi-squared Statistic: {test_df[CHI_SQUARED]}")
    print(f"P-value: {test_df[P_VALUE]}")
    print(f"Sample Size: {test_df[N]}")
    print(f"Degrees of Freedom: {test_df[DOF]}")
    print("Expected Frequencies:")
    print(test_df[FREQUENCIES])
    print("\n")


def do_cluster_by_trial_t_test(analysis_df: DataFrame, cluster_df: DataFrame):
    combs = list(itertools.combinations(cluster_df[CLUSTER].unique(), 2))
    t_tests_df = DataFrame()
    for cluster1, cluster2 in combs:
        trials1, trials2 = get_trials_by_cluster(cluster1, cluster_df, analysis_df), get_trials_by_cluster(cluster2, cluster_df, analysis_df)
        t_test_df = do_t_test_for_response_stats(trials1, trials2)
        t_test_df.index = [(cluster1, cluster2)]
        t_tests_df = concat([t_tests_df, t_test_df])

    t_test_df = correct_p_values(t_tests_df)
    return t_test_df


def do_lie_percentage_by_cluster_chi_squared_tests(analysis_df: DataFrame, cluster_df: DataFrame):
    combs = list(itertools.combinations(cluster_df[CLUSTER].unique(), 2))
    tests = {}
    for cluster1, cluster2 in combs:
        trials1, trials2 = get_trials_by_cluster(cluster1, cluster_df, analysis_df), get_trials_by_cluster(cluster2, cluster_df, analysis_df)
        tests[(cluster1, cluster2)] = do_lie_percentage_chi_squared_test(trials1, trials2, (cluster1, cluster2))

    tests_df = DataFrame.from_dict(tests, orient='index')

    tests_df = correct_p_values(tests_df)

    for label in tests_df.index:
        print_chi_squared_stats(tests_df.loc[label])


def do_lie_percentage_by_condition_chi_squared_tests(analysis_df: DataFrame, for_test: list[str]):

    for_test = [(GAIN_OF_TEN, GAIN_UNDER_TEN), (LOSS_OF_TEN, LOSS_UNDER_TEN), (GAIN_UNDER_THIRTY, GAIN_OF_THIRTY), (LOSS_UNDER_THIRTY, LOSS_OF_THIRTY)]
    tests = {}
    for condition1, condition2 in for_test:
        trials1, trials2 = get_trials_by_condition(condition1, analysis_df), get_trials_by_condition(condition2, analysis_df)
        tests[(condition1, condition2)] = do_lie_percentage_chi_squared_test(trials1, trials2, (condition1, condition2))

    tests_df = DataFrame.from_dict(tests, orient='index')

    tests_df = correct_p_values(tests_df)

    for label in tests_df.index:
        print_chi_squared_stats(tests_df.loc[label])


def do_lie_percentage_chi_squared_test(trials_df: DataFrame, other_trials_df: DataFrame, labels: list[str]):
    trials_is_lie = get_is_lie(trials_df).value_counts()
    other_trials_is_lie = get_is_lie(other_trials_df).value_counts()

    contingency_table = DataFrame(index=labels, data=[trials_is_lie, other_trials_is_lie])

    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    frequencies = DataFrame(expected, columns=contingency_table.columns, index=contingency_table.index)

    return Series(name=labels, data={N: len(trials_df + other_trials_df), CHI_SQUARED: chi2, P_VALUE: p, DOF: dof, FREQUENCIES: frequencies})


def do_anova_by_lie_percentage(analysis_df: DataFrame, cluster_df: DataFrame):
    trial_response_df = get_trial_response_stats_by_pid(cluster_df, analysis_df, with_error_bounds=False, val_only=True).reset_index()
    trial_response_df["TRIALCOUNT"] = trial_response_df[TRIAL_COUNT]

    model = ols(f'{LIE} ~ C({CLUSTER}) * TRIALCOUNT', data=trial_response_df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)
    display(anova_results, title=LIE)


def do_anova_by_variable(analysis_df: DataFrame, cluster_df: DataFrame):

    gain_of_ten_df = get_trials_by_condition(GAIN_OF_TEN, analysis_df)
    gain_of_less_than_ten_df = get_trials_by_condition(GAIN_UNDER_TEN, analysis_df)
    gain_of_ten_df[CLUSTER] = cluster_df[CLUSTER]
    gain_of_ten_df[CONDITION] = GAIN_OF_TEN
    gain_of_less_than_ten_df[CLUSTER] = cluster_df[CLUSTER]
    gain_of_less_than_ten_df[CONDITION] = GAIN_UNDER_TEN

    anova_df = concat([gain_of_ten_df.reset_index(), gain_of_less_than_ten_df.reset_index()])

    # model = ols(f'{LIE} ~ C({CONDITION}) * C({CLUSTER})', data=anova_df).fit()
    # anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    # display(anova_results, title=LIE)

    anova_df["SELFLIE"] = anova_df[SELF_LIE]
    anova_df["SELFTRUE"] = anova_df[SELF_TRUE]
    anova_df["OTHERLIE"] = anova_df[OTHER_LIE]
    anova_df["OTHERTRUTH"] = anova_df[OTHER_TRUTH]

    model = ols(f'SELFLIE ~ C({CONDITION}) * C({CLUSTER})', data=anova_df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    display(anova_results, title=SELF_LIE)

    model = ols(f'SELFTRUE ~ C({CONDITION}) * C({CLUSTER})', data=anova_df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    display(anova_results, title=SELF_TRUE)

    model = ols(f'OTHERLIE ~ C({CONDITION}) * C({CLUSTER})', data=anova_df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    display(anova_results, title=OTHER_LIE)

    model = ols(f'OTHERTRUTH ~ C({CONDITION}) * C({CLUSTER})', data=anova_df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    display(anova_results, title=OTHER_TRUTH)

    anova_df["NTRANSITIONS"] = anova_df[N_TRANSITIONS]
    model = ols(f'NTRANSITIONS ~ C({CONDITION}) * C({CLUSTER})', data=anova_df).fit()
    anova_results = sm.stats.anova_lm(model, typ=2)  # Type 2 ANOVA DataFrame
    display(anova_results, title=N_TRANSITIONS)

    maov = MANOVA.from_formula(f'SELFLIE + SELFTRUE + OTHERLIE + OTHERTRUTH + NTRANSITIONS ~ C({CONDITION}) * C({CLUSTER})', data=anova_df)
    print(maov.mv_test())


def plot_gains_by_trial_id(gains_df, to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    # plt.title('Net Gain to Sender by Trial ID')
    plt.ylabel('Net Gain to Sender')
    plt.bar(gains_df.index, gains_df[SELF_GAIN])
    plt.xticks(gains_df.index, fontsize=10)
    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def plot_losses_by_trial_id(gains_df, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.ylabel('Net Loss to Receiver')
    plt.bar(gains_df.index, gains_df[OTHER_LOSS], color=colors[0])
    plt.xticks(gains_df.index, fontsize=10)
    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()
