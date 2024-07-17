from typing import Callable
import numpy as np
from pandas import DataFrame
from sklearn.metrics import silhouette_score
from utils.display import display
from utils.paths import DTW_Z_V2_CSV
from utils.columns import *


def n_cluster_silhouette_analysis(aoi_df: DataFrame, get_clusters_fn: Callable[[DataFrame, int], DataFrame], metric: str = "euclidean", max_clusters: int = 20) -> DataFrame:
    silhouette_scores = {}
    for n in range(2, max_clusters + 1):
        cluster_df = get_clusters_fn(aoi_df, n_clusters=n)
        silhouette_scores[n] = {SILHOUETTE: silhouette_score(aoi_df, cluster_df[CLUSTER], metric=metric)}
    n_cluster_df = DataFrame.from_dict(silhouette_scores, orient="index")
    n_cluster_df.index.name = N_CLUSTER
    return n_cluster_df


def n_cluster_f_statistic_analysis(matrix_df: DataFrame, get_clusters_fn: Callable[[DataFrame, int], DataFrame], max_clusters: int = 20) -> DataFrame:
    n_cluster_start = 2
    pseudo_f = {}
    for n in range(n_cluster_start, max_clusters + 1):
        cluster_df = get_clusters_fn(matrix_df, n_clusters=n)
        pseudo_f[n] = {PSEUDO_F: pseudo_f_statistic(matrix_df, cluster_df)}
    n_cluster_df = DataFrame.from_dict(pseudo_f, orient="index")
    n_cluster_df.index.name = N_CLUSTER
    return n_cluster_df


def get_within_cluster_matrix(cluster, matrix_df, cluster_df):
    trial_1, trial_2 = matrix_df.index, matrix_df.columns
    cluster_trials = cluster_df[cluster_df[CLUSTER] == cluster]
    return matrix_df[trial_1.isin(cluster_trials.index)][trial_2.intersection(cluster_trials.index)]


def within_clusters(clusters, matrix_df, cluster_df, exclude_self_distances: bool = True):
    within_clusters = {}
    for cluster in clusters:
        within_matrix = get_within_cluster_matrix(cluster, matrix_df, cluster_df)
        within_vals = get_triangle_values(within_matrix, exclude_self_distances)
        within_clusters[cluster] = {MEAN: within_vals.mean(), VARIANCE: within_vals.var()}
    return DataFrame().from_dict(within_clusters, orient='index')


def get_between_cluster_matrix(cluster, matrix_df, cluster_df):
    trial_1, trial_2 = matrix_df.index, matrix_df.columns
    cluster_trials = cluster_df[cluster_df[CLUSTER] == cluster]
    return matrix_df[trial_1.isin(cluster_trials.index)][trial_2.difference(cluster_trials.index)]


def between_clusters(clusters, matrix_df, cluster_df):
    between_clusters = {}
    for cluster in clusters:
        between_matrix = get_between_cluster_matrix(cluster, matrix_df, cluster_df)
        between_vals = between_matrix.values
        between_clusters[cluster] = {MEAN: between_vals.mean(), VARIANCE: between_vals.var()}
    return DataFrame().from_dict(between_clusters, orient='index')


def pseudo_f_statistic(matrix_df: DataFrame, cluster_df: DataFrame, exclude_self_dist: bool = True):
    overall_size = len(matrix_df)
    clusters = cluster_df[CLUSTER].unique()
    n_clusters = len(clusters)
    within_deg_freedom = overall_size - n_clusters
    between_deg_freedom = n_clusters - 1
    within_cluster_df = within_clusters(clusters, matrix_df, cluster_df, exclude_self_dist)
    between_cluster_df = between_clusters(clusters, matrix_df, cluster_df)
    within_cluster_var = within_cluster_df[VARIANCE].sum()
    between_cluster_var = between_cluster_df[VARIANCE].sum()
    return (between_cluster_var / between_deg_freedom) / (within_cluster_var / within_deg_freedom)


def get_triangle_values(matrix_df: DataFrame, exclude_self_distances: bool = True):
    k = 1 if exclude_self_distances else 0
    return matrix_df.values[np.triu_indices_from(matrix_df.values, k=k)]


def get_best_fit_heirarchical_clusters(matrix_df: DataFrame, get_clusters_fn: Callable[[DataFrame, int], DataFrame], max_clusters: int = 20) -> DataFrame:
    n_cluster_df = n_cluster_f_statistic_analysis(matrix_df, get_clusters_fn, max_clusters)
    n_clusters = n_cluster_df.idxmax().values[0]
    return get_clusters_fn(matrix_df, n_clusters)


def get_best_fit_partitional_clusters(matrix_or_analysis_df: DataFrame, get_clusters_fn: Callable[[DataFrame, int], DataFrame], metric: str = 'euclidean', max_clusters: int = 20) -> DataFrame:
    n_cluster_df = n_cluster_silhouette_analysis(matrix_or_analysis_df, get_clusters_fn, metric, max_clusters)
    display(n_cluster_df)
    n_clusters = n_cluster_df.idxmax().values[0]
    return get_clusters_fn(matrix_or_analysis_df, n_clusters=n_clusters)


def get_best_fit_partitional_clusters_from_matrix(matrix_df: DataFrame, get_clusters_fn: Callable[[DataFrame, int], DataFrame], max_clusters: int = 20) -> DataFrame:
    return get_best_fit_partitional_clusters(matrix_df, get_clusters_fn, "precomputed", max_clusters)


def get_best_fit_partitional_clusters_from_features(analysis_df: DataFrame, get_clusters_fn: Callable[[DataFrame, int], DataFrame], metric: str = 'euclidean', max_clusters: int = 20) -> DataFrame:
    return get_best_fit_partitional_clusters(analysis_df, get_clusters_fn, "euclidean", max_clusters)
