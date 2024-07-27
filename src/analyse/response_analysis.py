
import os
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import XKCD_COLORS
import numpy as np
from pandas import DataFrame, Index
import seaborn as sns

from analyse.get_response_stats import correct_p_values, get_aggregate_cluster_t_test_response_stats, get_combination_t_test_response_stats, get_t_test_response_stats, get_response_stats, get_response_stats_for_clusters, get_response_stats_for_group_by
from utils.columns import CLUSTER, CLUSTER_1, CLUSTER_2, DISTANCE, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_OF_TWENTY, GAIN_UNDER_TEN, GROUP, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_OF_TWENTY, LOSS_UNDER_TEN, N_TRANSITIONS, NEGATIVE_GAIN, OTHER_LIE, OTHER_TRUTH, PID, PID_1, PID_2, POSITIVE_GAIN, RT, SELF_LIE, SELF_TRUE, TRIAL_COUNT, TRIAL_ID
from utils.display import display
from utils.masks import get_gain_of_between_ten_and_twenty, get_gain_of_between_twenty_and_thirty, get_gain_of_thirty, get_positive_gain
from utils.masks import get_gain_of_ten
from utils.masks import get_gain_of_twenty
from utils.masks import get_positive_gain_of_less_than_ten
from utils.masks import get_gain_of_less_than_ten
from utils.masks import get_negative_gain
from utils.masks import get_loss_of_ten
from utils.masks import get_loss_of_less_than_ten
from utils.masks import get_loss_of_thirty

XKCD_COLORS_LIST = list(XKCD_COLORS.values())


def calculate_mean_response_stat(df: DataFrame):
    mean = df.apply(lambda x: x[0]).mean()

    mean_lower_error = df.apply(lambda x: x[1][0]).mean()
    mean_upper_error = df.apply(lambda x: x[1][1]).mean()

    return mean, (mean_lower_error, mean_upper_error)


def get_trial_response_stats_by_pid(cluster_df: DataFrame, analysis_df: DataFrame):
    cluster_df.index = analysis_df.index
    response_df = get_response_stats_for_clusters(cluster_df.groupby([PID, CLUSTER]), analysis_df, analysis_df.index)
    return response_df


def get_response_stats_by_trial_id(cluster_df: DataFrame, analysis_df: DataFrame):
    by_trial_id_df = cluster_df.groupby(level=TRIAL_ID)
    response_df = get_response_stats_for_clusters(by_trial_id_df, analysis_df, analysis_df.index.get_level_values(TRIAL_ID))
    response_df[CLUSTER] = cluster_df[CLUSTER]
    return response_df.sort_values(CLUSTER)


def get_response_stats_by_pid(cluster_df: DataFrame, analysis_df: DataFrame):
    by_pid_df = analysis_df.groupby(PID)
    response_df = get_response_stats_for_group_by(by_pid_df)
    response_df[CLUSTER] = cluster_df[CLUSTER]
    return response_df.sort_values(CLUSTER)


def get_pid_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index.get_level_values(PID))


def get_trial_id_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index.get_level_values(TRIAL_ID))


def get_trial_count_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    analysis_df = analysis_df.reset_index().set_index([PID, TRIAL_COUNT])
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index.get_level_values(TRIAL_COUNT))


def get_trial_response_stats_for_clusters(cluster_df: DataFrame, analysis_df: DataFrame):
    cluster_df.index = analysis_df.index
    by_cluster_df = cluster_df.groupby(CLUSTER)
    return get_response_stats_for_clusters(by_cluster_df, analysis_df, analysis_df.index)


def get_trial_id_response_stats_no_clusters(analysis_df: DataFrame):
    return get_response_stats_for_group_by(analysis_df.groupby(TRIAL_ID))


def get_pid_response_stats_for_clusters_by_gain_label(cluster_df: DataFrame, gain_label: str, analysis_df: DataFrame):
    if gain_label == GAIN_UNDER_TEN:
        filtered_analysis_df = analysis_df.loc[get_gain_of_less_than_ten(analysis_df)]
    if gain_label == GAIN_OF_TEN:
        filtered_analysis_df = analysis_df.loc[get_gain_of_ten(analysis_df)]
    return get_pid_response_stats_for_clusters(cluster_df, filtered_analysis_df)


def get_pid_response_stats_no_clusters(analysis_df: DataFrame):
    return get_response_stats_for_group_by(analysis_df.groupby(PID))


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
    plt.title('%sPercent of Lies by TRIAL ID' % (title_prefix))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats(ax, responses_df, [LIE], colors=colors, to_file=to_file)


def plot_percent_lies_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%sPercent of Lies by PID' % (title_prefix))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [LIE], colors, to_file)


def plot_percent_lies_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sPercent of Lies per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [LIE], colors, to_file)


def plot_gain_of_ten_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(200, 6))
    plt.title('%sPercent Gain < 10 by PID' % (title_prefix))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [GAIN_OF_TEN], colors, to_file)


def plot_gain_under_ten_by_pid(responses_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(200, 6))
    plt.title('%sPercent Gain + 10 by PID' % (title_prefix))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters_by_pid(ax, responses_df, [GAIN_UNDER_TEN], colors, to_file)


def plot_gain_for_clusters(responses_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%sPercent Gain per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('Percent')
    plt.ylim(0, 100)
    plot_response_stats_for_clusters(ax, responses_df, [GAIN_OF_TEN, GAIN_UNDER_TEN], colors, to_file)


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


def plot_n_trials_for_clusters(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', sort_by: str = None, to_file: str = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('%sQuantity per %s Cluster' % (title_prefix, cluster_by))
    plt.ylabel('N Trials')
    response_df = response_df.sort_values(sort_by) if sort_by else response_df
    plot_response_stats_for_clusters(ax, response_df, [TRIAL_COUNT], colors, to_file)


def plot_n_trials_for_clusters_by_pid(response_df: DataFrame, cluster_by: str, colors: list[str] = XKCD_COLORS_LIST, title_prefix: str = '', title_override: str = None, to_file: str = None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title('%sQuantity per %s Cluster' % (title_prefix, cluster_by))
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
            ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[groups[group]], label=f'{group}')
        elif group_col == CLUSTER:
            cluster = int(enumerable_df.loc[value][CLUSTER])
            ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[cluster - 1], label=f'Cluster {cluster}')
        else:
            ax.bar(indices + i * bar_width, values, width=bar_width, color=colors[0])
        errors = [enumerable_df.loc[value][stat][1] for stat in stats]
        errors_y = create_error_bars_y(values, errors)
        ax.errorbar(indices + i * bar_width, values, errors_y, ecolor='black', capsize=5)

    plt.xticks([])
    if len(stats) > 1:
        plt.xticks(((bar_width/2 * len(responses_df.index)) + indices) - bar_width/2, stats, rotation=0)
    else:
        ax.set_xticks((bar_width/2) * (np.array(range(0, len(responses_df.index))) * 2))
        ax.set_xticklabels(responses_df.index, fontsize=7)

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
    ax.set_xticklabels(pids, fontsize=7)

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
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7])
    plt.ylabel('Frequency')
    plt.yticks([100, 200, 300, 400, 500])
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def plot_n_transitions_distributions(aoi_analysis_df: DataFrame, to_file: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(aoi_analysis_df[N_TRANSITIONS], kde=True)
    plt.title('Distribution of N Transitions')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()

def plot_rt_distributions(aoi_analysis_df: DataFrame, to_file: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(aoi_analysis_df[RT], kde=True)
    plt.title('Distribution of Reaction Times')
    plt.xlabel('Duration')
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


def plot_gains_mean_percent_lie(aoi_analysis_df, gain_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_to_show = []
    for gain in gain_labels:
        if gain == GAIN_OF_TEN:
            gain_of_ten_trials = aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_of_ten_trials)
            gains_to_show.append(response_df[LIE])
        if gain == GAIN_UNDER_TEN:
            gain_under_ten_trials = aoi_analysis_df.loc[get_positive_gain_of_less_than_ten(aoi_analysis_df)]
            response_df = get_response_stats(gain_under_ten_trials)
            gains_to_show.append(response_df[LIE])
        if gain == GAIN_OF_TWENTY in gain_labels:
            gain_of_twenty_trials = aoi_analysis_df.loc[get_gain_of_twenty(aoi_analysis_df)]
            response_df = get_response_stats(gain_of_twenty_trials)
            gains_to_show.append(response_df[LIE])
        if gain == GAIN_OF_THIRTY:
            gain_of_thirty_trials = aoi_analysis_df.loc[get_gain_of_thirty(aoi_analysis_df)]
            response_df = get_response_stats(gain_of_thirty_trials)
            gains_to_show.append(response_df[LIE])
        if gain == NEGATIVE_GAIN:
            negative_gain_trials = aoi_analysis_df.loc[get_negative_gain(aoi_analysis_df)]
            response_df = get_response_stats(negative_gain_trials)
            gains_to_show.append(response_df[LIE])
        if gain == POSITIVE_GAIN:
            positive_gain_trials = aoi_analysis_df.loc[get_positive_gain(aoi_analysis_df)]
            response_df = get_response_stats(positive_gain_trials)
            gains_to_show.append(response_df[LIE])
    simple_plot(gains_to_show, gain_labels, title="Percent of Lie based on Net Gain to Sender", ylabel="Percent Lie", colors=colors, to_file=to_file)


def plot_gains_avg_dwell_time(analysis_df: DataFrame, gain_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([GAIN_UNDER_TEN, GAIN_OF_TEN, POSITIVE_GAIN, NEGATIVE_GAIN], name=GROUP), columns=[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH])
    gain_of_ten_trials = analysis_df.loc[get_gain_of_ten(analysis_df)]
    gain_under_ten_trials = analysis_df.loc[get_positive_gain_of_less_than_ten(analysis_df)]
    positive_gain_trials = analysis_df.loc[get_positive_gain(analysis_df)]
    negative_gain_trials = analysis_df.loc[get_negative_gain(analysis_df)]
    gains_df.loc[GAIN_OF_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(gain_of_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[GAIN_UNDER_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(gain_under_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[POSITIVE_GAIN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(positive_gain_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[NEGATIVE_GAIN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(negative_gain_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]

    fig, ax = plt.subplots(figsize=(25, 6))
    plt.title('Average Dwell Time for each AOI per differing quantities of net gain to sender')
    plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_groups(ax, gains_df.loc[gain_labels], [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH], colors, to_file)


def plot_gains_n_transitions(analysis_df: DataFrame, gain_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([GAIN_UNDER_TEN, GAIN_OF_TEN, POSITIVE_GAIN, NEGATIVE_GAIN], name=GROUP), columns=[N_TRANSITIONS])
    gain_of_ten_trials = analysis_df.loc[get_gain_of_ten(analysis_df)]
    gain_under_ten_trials = analysis_df.loc[get_positive_gain_of_less_than_ten(analysis_df)]
    positive_gain_trials = analysis_df.loc[get_positive_gain(analysis_df)]
    negative_gain_trials = analysis_df.loc[get_negative_gain(analysis_df)]
    gains_df = DataFrame(index=Index([GAIN_UNDER_TEN, GAIN_OF_TEN], name=GROUP), columns=[N_TRANSITIONS])
    gains_df.loc[GAIN_OF_TEN, N_TRANSITIONS] = get_response_stats(gain_of_ten_trials)[N_TRANSITIONS]
    gains_df.loc[GAIN_UNDER_TEN, N_TRANSITIONS] = get_response_stats(gain_under_ten_trials)[N_TRANSITIONS]
    gains_df.loc[POSITIVE_GAIN, N_TRANSITIONS] = get_response_stats(positive_gain_trials)[N_TRANSITIONS]
    gains_df.loc[NEGATIVE_GAIN, N_TRANSITIONS] = get_response_stats(negative_gain_trials)[N_TRANSITIONS]

    fig, ax = plt.subplots(figsize=(25, 6))
    plt.title('Average N Transitions per differing quantities of net gain to sender')
    plt.ylabel('N Transitions')
    plot_response_stats_for_groups(ax, gains_df.loc[gain_labels], [N_TRANSITIONS], colors, to_file)


def plot_losses_avg_dwell_time(analysis_df: DataFrame, loss_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([LOSS_UNDER_TEN, LOSS_OF_TEN], name=GROUP), columns=[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH])
    loss_of_ten_trials = analysis_df.loc[get_loss_of_ten(analysis_df)]
    loss_under_ten_trials = analysis_df.loc[get_loss_of_less_than_ten(analysis_df)]
    gains_df.loc[LOSS_OF_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(loss_of_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]
    gains_df.loc[LOSS_UNDER_TEN, [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]] = get_response_stats(loss_under_ten_trials)[[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]]

    fig, ax = plt.subplots(figsize=(25, 6))
    plt.title('Average Dwell Time for each AOI per differing quantities of net loss to reciever')
    plt.ylabel('Dwell Time (ms)')
    plot_response_stats_for_groups(ax, gains_df.loc[loss_labels], [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH], colors, to_file)


def plot_losses_n_transitions(analysis_df: DataFrame, loss_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    gains_df = DataFrame(index=Index([LOSS_UNDER_TEN, LOSS_OF_TEN], name=GROUP), columns=[N_TRANSITIONS])
    loss_of_ten_trials = analysis_df.loc[get_loss_of_ten(analysis_df)]
    loss_under_ten_trials = analysis_df.loc[get_loss_of_less_than_ten(analysis_df)]
    gains_df.loc[LOSS_OF_TEN, N_TRANSITIONS] = get_response_stats(loss_of_ten_trials)[N_TRANSITIONS]
    gains_df.loc[LOSS_UNDER_TEN, N_TRANSITIONS] = get_response_stats(loss_under_ten_trials)[N_TRANSITIONS]

    fig, ax = plt.subplots(figsize=(25, 6))
    plt.title('Average N Transitions per differing quantities of net loss to reciever')
    plt.ylabel('N Transitions')
    plot_response_stats_for_groups(ax, gains_df.loc[loss_labels], [N_TRANSITIONS], colors, to_file)


def plot_losses_mean_percent_lie(aoi_analysis_df: DataFrame, loss_labels: list[str], colors: list[str] = XKCD_COLORS_LIST, to_file=None):
    losses_to_show = []
    for loss in loss_labels:
        if loss == LOSS_OF_TEN:
            loss_of_ten_trials = aoi_analysis_df.loc[get_gain_of_ten(aoi_analysis_df)]
            response_df = get_response_stats(loss_of_ten_trials)
            losses_to_show.append(response_df[LIE])
        if loss == LOSS_UNDER_TEN:
            loss_under_ten = aoi_analysis_df.loc[get_loss_of_less_than_ten(aoi_analysis_df)]
            response_df = get_response_stats(loss_under_ten)
            losses_to_show.append(response_df[LIE])
        if loss == LOSS_OF_TWENTY:
            loss_of_twenty = aoi_analysis_df.loc[get_gain_of_twenty(aoi_analysis_df)]
            response_df = get_response_stats(loss_of_twenty)
            losses_to_show.append(response_df[LIE])
        if loss == LOSS_OF_THIRTY:
            loss_of_thirty = aoi_analysis_df.loc[get_loss_of_thirty(aoi_analysis_df)]
            response_df = get_response_stats(loss_of_thirty)
            losses_to_show.append(response_df[LIE])
    simple_plot(losses_to_show, loss_labels, title="Percent of Lie based on Net Loss to Receiver", ylabel="Percent Lie", colors=colors, to_file=to_file)


def simple_plot(x, xlabel, title, ylabel, yticks=[], colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    vals = [val[0] for val in x]
    errors = [val[1] for val in x]
    yerr = create_error_bars_y(vals, errors)
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.title(title)
    plt.ylabel(ylabel)
    plt.bar(xlabel, vals, yerr=yerr, capsize=5, color=colors)
    plt.xticks(xlabel, fontsize=7)
    plt.tight_layout()
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()


def do_gains_t_test(analysis_df: DataFrame):
    gain_of_ten_trials = analysis_df.loc[get_gain_of_ten(analysis_df)]
    gain_under_ten_trials = analysis_df.loc[get_positive_gain_of_less_than_ten(analysis_df)]
    between_ten_twenty = analysis_df.loc[get_gain_of_between_ten_and_twenty(analysis_df)]
    gain_of_twenty_trials = analysis_df.loc[get_gain_of_twenty(analysis_df)]
    gain_of_thirty_trials = analysis_df.loc[get_gain_of_thirty(analysis_df)]
    between_twenty_thirty = analysis_df.loc[get_gain_of_between_twenty_and_thirty(analysis_df)]
    positive_gain_trials = analysis_df.loc[get_positive_gain(analysis_df)]
    negative_gain_trials = analysis_df.loc[get_negative_gain(analysis_df)]

    ps_to_correct = {}

    ps_to_correct[(GAIN_OF_TEN, GAIN_UNDER_TEN)] = get_t_test_response_stats(gain_of_ten_trials, gain_under_ten_trials)

    ps_to_correct[(POSITIVE_GAIN, NEGATIVE_GAIN)] = get_t_test_response_stats(positive_gain_trials, negative_gain_trials)

    get_t_test_response_stats(gain_of_twenty_trials, gain_of_thirty_trials)
    get_t_test_response_stats(between_ten_twenty, between_twenty_thirty)
    get_t_test_response_stats(gain_under_ten_trials, between_ten_twenty)

    t_test_df = correct_p_values(DataFrame.from_dict(ps_to_correct, orient='index'))

    return t_test_df

def do_clustered_pid_t_test(analysis_df: DataFrame, cluster_df: DataFrame):

    group_by_pid = analysis_df.groupby(PID)

    comb_t_tests = get_combination_t_test_response_stats(group_by_pid)
    comb_t_tests.index.set_names([PID_1, PID_2], inplace=True)
    comb_t_tests[CLUSTER_1] = cluster_df.loc[comb_t_tests.index.get_level_values(PID_1), CLUSTER].values
    comb_t_tests[CLUSTER_2] = cluster_df.loc[comb_t_tests.index.get_level_values(PID_2), CLUSTER].values
    
    return get_aggregate_cluster_t_test_response_stats(comb_t_tests)
