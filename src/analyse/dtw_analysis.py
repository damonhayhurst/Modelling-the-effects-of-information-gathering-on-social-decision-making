from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.colors import XKCD_COLORS, LogNorm
from pandas import DataFrame, MultiIndex, concat
from utils.display import display
from utils.paths import PID_DISTANCE_PLOT, TRIAL_DISTANCE_PLOT
from utils.masks import is_same_pid, is_same_trial, is_selected_aoi_same
from utils.columns import *
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

XKCD_COLORS_LIST = list(XKCD_COLORS.values())


def print_aoi_stats(distance_df: DataFrame):
    avg_distance = get_overall_avg_distance(distance_df)
    print("Mean Distance: %s" % avg_distance)
    is_same_aoi = is_selected_aoi_same(distance_df, SELECTED_AOI_1, SELECTED_AOI_2)
    same_aoi_distance = distance_df[is_same_aoi][DISTANCE].mean()
    diff_aoi_distance = distance_df[~is_same_aoi][DISTANCE].mean()
    print("Mean Distance for same response: %s" % same_aoi_distance)
    print("Mean Distance for differing responses: %s" % diff_aoi_distance)


def get_overall_avg_distance(distance_df: DataFrame):
    return distance_df[DISTANCE].mean()


def create_big_matrix(distance_df: DataFrame):
    distance_df = distance_df[DISTANCE].to_frame()
    distance_df.replace(np.inf, np.nan, inplace=True)
    distance_df.loc[is_same_trial(distance_df), DISTANCE] = 0
    swapped_df = distance_df.swaplevel(PID_1, PID_2).swaplevel(TRIAL_ID_1, TRIAL_ID_2).swaplevel(TRIAL_COUNT_1, TRIAL_COUNT_2)
    swapped_df.index.names = ([PID_1, TRIAL_ID_1, TRIAL_COUNT_1, PID_2, TRIAL_ID_2, TRIAL_COUNT_2])
    distance_df[DISTANCE] = distance_df[DISTANCE].fillna(swapped_df[DISTANCE])
    return distance_df.unstack([PID_2, TRIAL_ID_2, TRIAL_COUNT_2]).sort_values([PID_1, TRIAL_ID_1, TRIAL_COUNT_1]).sort_values([PID_2, TRIAL_ID_2], axis=1)


def is_same_trial(distance_df: DataFrame):
    return (distance_df.index.get_level_values(0) == distance_df.index.get_level_values(2)) & \
        (distance_df.index.get_level_values(1) == distance_df.index.get_level_values(3))


def print_pid_stats(distance_df: DataFrame):
    same_pid = is_same_pid(distance_df, PID_1, PID_2)
    same_pid_distance = distance_df[same_pid][DISTANCE].mean()
    diff_pid_distance = distance_df[~same_pid][DISTANCE].mean()
    print("Mean Distance for same pids: %s" % same_pid_distance)
    print("Mean Distance for different pids: %s" % diff_pid_distance)


def create_matrix(big_matrix_df: DataFrame, x_idx: str, y_idx: str, fill_self_distances: int | None = 0):
    by_trial = big_matrix_df.T.groupby(y_idx).mean()
    matrix_df = by_trial.T.groupby(x_idx).mean()
    if fill_self_distances is not None:
        matrix_df = set_diagonal(matrix_df, fill_self_distances)
    return matrix_df


def create_pid_matrix(big_matrix_df: DataFrame, fill_self_distances: int | None = 0):
    return create_matrix(big_matrix_df, PID_1, PID_2, fill_self_distances)


def plot_pid_matrix_with_clusters(matrix_df: DataFrame, cluster_df: DataFrame, colors: list[str] = list(XKCD_COLORS.values()), to_file: str = None):
    plot_matrix_with_heirarchical_clusters(matrix_df, cluster_df, index_name=PID, colors=colors, to_file=to_file)


def print_trial_id_stats(distance_df: DataFrame):
    same_trial = is_same_trial(distance_df, TRIAL_ID_1, TRIAL_ID_2)
    same_trial_distance = distance_df[same_trial][DISTANCE].mean()
    diff_trial_distance = distance_df[~same_trial][DISTANCE].mean()
    print("Mean Distance for same trial: %s" % same_trial_distance)
    print("Mean Distance for different trials: %s" % diff_trial_distance)
    return distance_df


def create_trial_id_matrix(big_matrix_df: DataFrame, fill_self_distances: int | None = 0):
    return create_matrix(big_matrix_df, TRIAL_ID_1, TRIAL_ID_2, fill_self_distances)


def create_trial_count_matrix(big_matrix_df: DataFrame, fill_self_distances: int | None = 0):
    return create_matrix(big_matrix_df, TRIAL_COUNT_1, TRIAL_COUNT_2, fill_self_distances)

def set_diagonal(matrix_df: DataFrame, value: int = 0) -> DataFrame:
    matrix = matrix_df.values
    np.fill_diagonal(matrix, value)
    return DataFrame(matrix, index=matrix_df.index, columns=matrix_df.columns)


def plot_trial_id_matrix_with_clusters(matrix_df: DataFrame, cluster_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    plot_matrix_with_heirarchical_clusters(matrix_df, cluster_df, index_name=TRIAL_ID, colors=colors, to_file=to_file)


def plot_trial_count_matrix_with_clusters(matrix_df: DataFrame, cluster_df: DataFrame, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    plot_matrix_with_heirarchical_clusters(matrix_df, cluster_df, index_name=TRIAL_COUNT, colors=colors, to_file=to_file)

def plot_matrix_with_heirarchical_clusters(matrix_df: DataFrame, cluster_df: DataFrame, index_name: str, colors: list[str] = XKCD_COLORS_LIST, to_file: str = None):
    matrix_df = reorder_matrix_by_heirarchical_cluster(matrix_df, cluster_df)

    plt.figure(figsize=(12, 10))
    heatmap = sns.heatmap(matrix_df, cmap='coolwarm', norm=LogNorm(), cbar=True, linewidths=0)

    def add_cluster_boundaries(ax: Axes, cluster_df: DataFrame, colors=colors, linewidth=2):
        clusters = cluster_df[CLUSTER].unique()
        colors = colors[:len(clusters) + 1]
        for cluster, color in zip(clusters, colors):
            indices = np.where(cluster_df == cluster)[0]
            min_idx, max_idx = min(indices), max(indices)
            rect = Rectangle((min_idx, min_idx), max_idx - min_idx + 1, max_idx - min_idx + 1, fill=False, edgecolor=color, lw=linewidth)
            ax.add_patch(rect)

    # Add cluster boundaries
    add_cluster_boundaries(heatmap, cluster_df, colors)
    plt.gca().invert_yaxis()
    plt.xticks(ticks=np.arange(len(matrix_df.columns)) + 0.5, labels=matrix_df.columns, fontsize=6)
    plt.yticks(ticks=np.arange(len(matrix_df.index)) + 0.5, labels=matrix_df.index, rotation=360, fontsize=6)
    plt.xlabel(index_name)
    plt.ylabel(index_name)
    plt.tick_params(left=False, bottom=False)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    if to_file is not None:
        plt.savefig(to_file)
    plt.show()


def get_heirarchical_clusters_by_trial_id(matrix_df: DataFrame, n_clusters: int = 3):
    return get_heirarchical_clusters(matrix_df, TRIAL_ID, n_clusters)


def get_heirarchical_clusters_by_pid(matrix_df: DataFrame, n_clusters: int = 3):
    return get_heirarchical_clusters(matrix_df, PID, n_clusters)


def get_heirarchical_clusters_by_trial_count(matrix_df: DataFrame, n_clusters: int = 3):
    return get_heirarchical_clusters(matrix_df, TRIAL_COUNT, n_clusters)

def get_heirarchical_clusters(matrix_df: DataFrame, index_name: str, n_clusters: int = 3):
    row_clusters = linkage(matrix_df, method='complete', metric='euclidean')
    cluster_labels = fcluster(row_clusters, t=n_clusters, criterion='maxclust')
    return DataFrame({
        CLUSTER: cluster_labels
    }, index=matrix_df.index.set_names(index_name)).sort_values(CLUSTER)


def reorder_matrix_by_heirarchical_cluster(matrix_df: DataFrame, cluster_df: DataFrame):
    return matrix_df.loc[cluster_df.index, :].loc[:, cluster_df.index]
