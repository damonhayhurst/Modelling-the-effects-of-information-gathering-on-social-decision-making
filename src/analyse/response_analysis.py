
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from pandas import DataFrame

from analyse.dtw_analysis import XKCD_COLORS_LIST
from utils.columns import CLUSTER, LIE, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, OTHER_LIE, OTHER_TRUTH, PID, SELECTED_AOI, SELF_LIE, SELF_TRUE, TRIAL_ID
from utils.display import display

def get_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame, index_name: str):

    def percent_lie(df: DataFrame):
        cluster_trials = analysis_df.loc[analysis_df.index.get_level_values(index_name).isin(df.index)]
        is_truth = cluster_trials[SELECTED_AOI] == LIE
        return is_truth.mean() * 100

    def n_transitions(df: DataFrame):
        cluster_trials = analysis_df.loc[analysis_df.index.get_level_values(index_name).isin(df.index)]
        return (cluster_trials[N_ALT_TRANSITIONS] + cluster_trials[N_ATT_TRANSITIONS]).mean()

    def avg_dwell_time(df: DataFrame, aoi: str):
        cluster_trials = analysis_df.loc[analysis_df.index.get_level_values(index_name).isin(df.index)]
        return cluster_trials[aoi].mean()

    return DataFrame({
        LIE: cluster_df.groupby(CLUSTER).apply(percent_lie),
        N_TRANSITIONS: cluster_df.groupby(CLUSTER).apply(n_transitions),
        SELF_LIE: cluster_df.groupby(CLUSTER).apply(avg_dwell_time, SELF_LIE),
        SELF_TRUE: cluster_df.groupby(CLUSTER).apply(avg_dwell_time, SELF_TRUE),
        OTHER_LIE: cluster_df.groupby(CLUSTER).apply(avg_dwell_time, OTHER_LIE),
        OTHER_TRUTH: cluster_df.groupby(CLUSTER).apply(avg_dwell_time, OTHER_TRUTH)
    })


def get_pid_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    return get_response_stats_for_clusters(cluster_df, analysis_df, PID)


def get_trial_id_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    return get_response_stats_for_clusters(cluster_df, analysis_df, TRIAL_ID)


def plot_percent_lies_for_clusters(responses_df: DataFrame, index_name: str, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Percent of Lies per %s Cluster' % index_name)
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [LIE], index_name, colors, to_file)


def plot_dwell_times_for_clusters(responses_df: DataFrame, index_name: str, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Average Dwell Time for each AOI per %s Cluster' % index_name)
    plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_clusters(ax, responses_df, [SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH], index_name, colors, to_file)


def plot_n_transitions_for_clusters(response_df: DataFrame, index_name: str, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Number of Transitions per %s Cluster' % index_name)
    plt.ylabel('N Transitions')
    plot_response_stats_for_clusters(ax, response_df, [N_TRANSITIONS], index_name, colors, to_file)

def plot_response_stats_for_clusters(ax: Axes, responses_df: DataFrame, stats: list[str], index_name: str, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):

    num_categories = len(stats)
    bar_width = 0.2

    indices = np.arange(num_categories)
    for stat in stats:
        for i, cluster in enumerate(responses_df.index):
            values = [responses_df.loc[cluster][stat] for stat in stats]
            ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[i], label=f'Cluster {cluster}')

    plt.xticks([])
    if len(stats) > 1:
        plt.xticks((bar_width/2) + indices, stats, rotation=0)

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
