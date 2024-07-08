
import os
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from pandas import DataFrame, Index
import seaborn as sns

from analyse.dtw_analysis import XKCD_COLORS_LIST
from utils.columns import CLUSTER, LIE, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, OTHER_LIE, OTHER_TRUTH, PID, SELECTED_AOI, SELF_LIE, SELF_TRUE, TRIAL_COUNT, TRIAL_ID
from utils.display import display


def get_response_stats(cluster_df: DataFrame, analysis_df: DataFrame, index: Index):

    def percent_lie(df: DataFrame):
        cluster_trials = analysis_df.loc[index.isin(df.index)]
        is_truth = cluster_trials[SELECTED_AOI] == LIE
        return is_truth.mean() * 100, is_truth.std() * 100

    def n_transitions(df: DataFrame):
        cluster_trials = analysis_df.loc[index.isin(df.index)]
        mean_transitions = (cluster_trials[N_ALT_TRANSITIONS] + cluster_trials[N_ATT_TRANSITIONS]).mean()
        std_transitions = (cluster_trials[N_ALT_TRANSITIONS] + cluster_trials[N_ATT_TRANSITIONS]).std()
        return np.round(mean_transitions), np.round(std_transitions)

    def avg_dwell_time(df: DataFrame, aoi: str):
        cluster_trials = analysis_df.loc[index.isin(df.index)]
        return cluster_trials[aoi].mean(), cluster_trials[aoi].std()

    def n_trials(df: DataFrame):
        cluster_trials = analysis_df.loc[index.isin(df.index)]
        return len(cluster_trials)

    def calc_percent_error_bounds(df: DataFrame):
        val, err = df
        upper_err, lower_err = min(val + err/2, 100), max(val - err/2, 0)
        return val, (lower_err, upper_err)

    def calc_duration_error_bounds(df: DataFrame):
        val, err = df
        upper_err, lower_err = val + err/2, max(val - err/2, 0)
        return val, (lower_err, upper_err)

    def calc_count_error_bounds(df: DataFrame):
        val, err = df[0], df[1]
        upper_err, lower_err = val + err/2, max(val - err/2, 0)
        upper_err = upper_err if np.isnan(upper_err) else int(upper_err)
        lower_err = lower_err if np.isnan(lower_err) else int(lower_err)
        return val, (lower_err, upper_err)

    def no_error_bounds(df: DataFrame):
        return df, (np.nan, np.nan)


    return DataFrame({
        LIE: cluster_df.apply(percent_lie).apply(calc_percent_error_bounds),
        N_TRANSITIONS: cluster_df.apply(n_transitions).apply(calc_count_error_bounds),
        SELF_LIE: cluster_df.apply(avg_dwell_time, aoi=SELF_LIE).apply(calc_duration_error_bounds),
        SELF_TRUE: cluster_df.apply(avg_dwell_time, aoi=SELF_TRUE).apply(calc_duration_error_bounds),
        OTHER_LIE: cluster_df.apply(avg_dwell_time, aoi=OTHER_LIE).apply(calc_duration_error_bounds),
        OTHER_TRUTH: cluster_df.apply(avg_dwell_time, aoi=OTHER_TRUTH).apply(calc_duration_error_bounds),
        TRIAL_COUNT: cluster_df.apply(n_trials).apply(no_error_bounds),
    })


def get_trial_response_stats_by_pid(cluster_df: DataFrame, analysis_df: DataFrame):
    cluster_df.index = analysis_df.index
    response_df = get_response_stats(cluster_df.groupby([PID, CLUSTER]), analysis_df, analysis_df.index)
    return response_df


def get_response_stats_by_trial_id(cluster_df: DataFrame, analysis_df: DataFrame):
    by_trial_id_df = cluster_df.groupby(level=TRIAL_ID)
    response_df = get_response_stats(by_trial_id_df, analysis_df, analysis_df.index.get_level_values(TRIAL_ID))
    response_df[CLUSTER] = cluster_df[CLUSTER]
    return response_df.sort_values(CLUSTER)


def get_response_stats_by_pid(cluster_df: DataFrame, analysis_df: DataFrame):
    by_pid_df = cluster_df.groupby(level=PID)
    response_df = get_response_stats(by_pid_df, analysis_df, analysis_df.index.get_level_values(PID))
    response_df[CLUSTER] = cluster_df[CLUSTER]
    return response_df.sort_values(CLUSTER)


def get_pid_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats(by_cluster_df, analysis_df, analysis_df.index.get_level_values(PID))


def get_trial_id_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats(by_cluster_df, analysis_df, analysis_df.index.get_level_values(TRIAL_ID))


def get_trial_count_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    analysis_df = analysis_df.reset_index().set_index([PID, TRIAL_COUNT])
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats(by_cluster_df, analysis_df, analysis_df.index.get_level_values(TRIAL_COUNT))


def get_trial_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    cluster_df.index = analysis_df.index
    display(cluster_df)
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats(by_cluster_df, analysis_df, analysis_df.index)


def get_response_stats_no_clusters(analysis_df: DataFrame):
    return get_response_stats(analysis_df.groupby(PID), analysis_df, analysis_df.index)

def get_trials_by_cluster(cluster: int, cluster_df: DataFrame, analysis_df: DataFrame):
    cluster_df.index = analysis_df.index
    by_cluster_df = cluster_df.groupby(CLUSTER)
    idx = by_cluster_df.get_group(cluster).index
    return analysis_df.loc[analysis_df.index.isin(idx)]


def plot_percent_lies_by_trial_id(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.title('%sPercent of Lies by TRIAL ID' % (title_prefix))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [LIE], colors, to_file)


def plot_percent_lies_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%sPercent of Lies by PID' % (title_prefix))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    display(responses_df[LIE], max_rows=None)
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [LIE], colors, to_file)


def plot_percent_lies_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sPercent of Lies per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [LIE], colors, to_file)


def plot_dwell_times_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%sAverage Dwell Time for each AOI per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_clusters(ax, responses_df, [SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH], colors, to_file)


def plot_n_transitions_for_clusters(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sNumber of Transitions per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('N Transitions')
    plot_response_stats_for_clusters(ax, response_df, [N_TRANSITIONS], colors, to_file)


def plot_n_trials_for_clusters(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sQuantity per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('N Trials')
    plot_response_stats_for_clusters(ax, response_df, [TRIAL_COUNT], colors, to_file)


def plot_n_trials_for_clusters_by_pid(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%sQuantity per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('N Trials')
    plot_response_stats_for_clusters_by_pid(ax, response_df, [TRIAL_COUNT], colors, to_file)

def create_error_bars_y(values, errors):
    error_bars = []
    for v in range(len(values)):
        value = values[v]
        lower, upper = errors[v][0], errors[v][1]
        error_bars.append([value - lower, upper - value])
    return np.array(error_bars).T


def plot_response_stats_for_clusters(ax: Axes, responses_df: DataFrame, stats: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    display(responses_df)
    num_categories = len(stats)
    bar_width = 0.3
    indices = np.arange(num_categories)
    enumerable_df = responses_df.reset_index()
    is_clustered = CLUSTER in enumerable_df.columns

    for i, value in enumerate(enumerable_df.index):
        values = [enumerable_df.loc[value][stat][0] for stat in stats]
        cluster = int(enumerable_df.loc[value][CLUSTER]) if is_clustered else 0
        ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[cluster - 1], label=f'Cluster {cluster}')
        errors = [enumerable_df.loc[value][stat][1] for stat in stats]
        errors_y = create_error_bars_y(values, errors)
        ax.errorbar(indices + i * bar_width, values, errors_y, ecolor='black', capsize=5)

    plt.xticks([])
    if len(stats) > 1:
        plt.xticks(((bar_width/2 * len(responses_df.index)) + indices) - bar_width/2, stats, rotation=0)

    if is_clustered:
        plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2)

    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)

    plt.show()


def plot_response_stats_for_clusters_by_pid(ax: Axes, responses_df: DataFrame, stats: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    bar_width = 0.3
    gap_width = 0.45  # Width of the gap between different PIDs
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

    plt.xticks([])
    if len(stats) > 1:
        plt.xticks(((bar_width/2 * len(responses_df.index)) + np.arange(len(pids) * (num_categories + 1))) - bar_width/2, stats, rotation=0)

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
    ax.set_xticklabels(pids)

    ax.tick_params(axis='both', which='both', length=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.tick_params(width=1.2)

    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)

    plt.show()






def plot_dwell_time_distributions(aoi_analysis_df: DataFrame, aoi: str, to_file: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(aoi_analysis_df[aoi], kde=True)
    plt.title('Distribution of %s Dwell Times' % aoi)
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()
