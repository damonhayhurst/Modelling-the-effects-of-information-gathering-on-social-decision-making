
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from pandas import DataFrame, Index

from analyse.dtw_analysis import XKCD_COLORS_LIST
from utils.columns import CLUSTER, LIE, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, OTHER_LIE, OTHER_TRUTH, PID, SELECTED_AOI, SELF_LIE, SELF_TRUE, TRIAL_COUNT, TRIAL_ID
from utils.display import display


def get_response_stats(cluster_df: DataFrame, analysis_df: DataFrame, index: Index):

    def percent_lie(df: DataFrame):
        cluster_trials = analysis_df.loc[index.isin(df.index)]
        is_truth = cluster_trials[SELECTED_AOI] == LIE
        return is_truth.mean() * 100

    def n_transitions(df: DataFrame):
        cluster_trials = analysis_df.loc[index.isin(df.index)]
        return (cluster_trials[N_ALT_TRANSITIONS] + cluster_trials[N_ATT_TRANSITIONS]).mean()

    def avg_dwell_time(df: DataFrame, aoi: str):
        cluster_trials = analysis_df.loc[index.isin(df.index)]
        return cluster_trials[aoi].mean()

    return DataFrame({
        LIE: cluster_df.apply(percent_lie),
        N_TRANSITIONS: cluster_df.apply(n_transitions),
        SELF_LIE: cluster_df.apply(avg_dwell_time, aoi=SELF_LIE),
        SELF_TRUE: cluster_df.apply(avg_dwell_time, aoi=SELF_TRUE),
        OTHER_LIE: cluster_df.apply(avg_dwell_time, aoi=OTHER_LIE),
        OTHER_TRUTH: cluster_df.apply(avg_dwell_time, aoi=OTHER_TRUTH)
    })


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
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats(by_cluster_df, analysis_df, analysis_df.index)


def plot_percent_lies_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sPercent of Lies by PID' % (title_prefix))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [LIE], PID, colors, to_file)

def plot_percent_lies_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sPercent of Lies per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [LIE], CLUSTER, colors, to_file)


def plot_dwell_times_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sAverage Dwell Time for each AOI per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_clusters(ax, responses_df, [SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH], CLUSTER, colors, to_file)


def plot_n_transitions_for_clusters(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sNumber of Transitions per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('N Transitions')
    plot_response_stats_for_clusters(ax, response_df, [N_TRANSITIONS], CLUSTER, colors, to_file)


def plot_response_stats_for_clusters(ax: Axes, responses_df: DataFrame, stats: list[str], index_name: str, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):

    num_categories = len(stats)
    bar_width = 0.3

    indices = np.arange(num_categories)
    enumerable_df = responses_df.reset_index()
    for i, value in enumerate(enumerable_df.index):
        values = [enumerable_df.loc[value][stat] for stat in stats]
        cluster = int(enumerable_df.loc[value][CLUSTER])
        ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[cluster], label=f'Cluster {cluster}')

    plt.xticks([])
    if len(stats) > 1 and index_name is CLUSTER:
        plt.xticks((bar_width/2) + indices, stats, rotation=0)
    elif index_name is PID:
        plt.xticks((bar_width/2) * (np.array(range(0, len(responses_df.index))) * 2), responses_df.index, rotation=0, fontsize=8)

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
        plt.savefig(to_file)

    plt.show()
