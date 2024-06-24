from typing import List
from pandas import DataFrame
from analyse.dtw_analysis import XKCD_COLORS_LIST, create_big_matrix, create_pid_matrix, create_trial_count_matrix, create_trial_id_matrix, get_heirarchical_clusters_by_pid, get_heirarchical_clusters_by_trial, get_heirarchical_clusters_by_trial_count, get_heirarchical_clusters_by_trial_id, plot_distance_distribution, plot_pid_matrix_with_clusters, plot_trial_count_matrix_with_clusters, plot_trial_id_matrix_with_clusters, get_proximal_and_distal_distances, set_diagonal
from analyse.kmeans_analysis import cluster_analysis, get_best_fit_kmeans_clusters, get_kmeans_clusters, merge_components, prepare_data
from analyse.n_cluster_analysis import get_best_fit_heirarchical_clusters
from analyse.cluster_response_analysis import get_pid_response_stats_for_clusters, get_response_stats_by_pid, get_response_stats_by_trial_id, get_trial_count_response_stats_for_clusters, get_trial_id_response_stats_for_clusters, get_trial_response_stats_for_clusters, plot_dwell_times_for_clusters, plot_n_transitions_for_clusters, plot_percent_lies_by_pid, plot_percent_lies_by_trial_id, plot_percent_lies_for_clusters, plot_response_stats_for_clusters
from analyse.response_analysis import plot_dwell_time_distributions
from utils.columns import CLUSTER, DISTANCE, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, OTHER_LIE, OTHER_TRUTH, PAYNE_INDEX, PID, SELF_LIE, SELF_TRUE, TRIAL, TRIAL_COUNT, TRIAL_ID
from utils.display import display
from utils.masks import is_no_dwell_for_aois
from utils.paths import *
from utils.read_csv import read_from_analysis_file, read_from_dtw_file


COLORS = XKCD_COLORS_LIST


def response_analysis(input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                      self_lie_distribution_plot: str = SELF_LIE_DISTRIBUTION_PLOT,
                      self_true_distribution_plot: str = SELF_TRUE_DISTRIBUTION_PLOT,
                      other_lie_distribution_plot: str = OTHER_LIE_DISTRIBUTION_PLOT,
                      other_true_distribution_plot: str = OTHER_TRUTH_DISTRIBUTION_PLOT):
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    is_no_dwell = is_no_dwell_for_aois(aoi_analysis_df, [SELF_TRUE, SELF_LIE, OTHER_TRUTH, OTHER_LIE])
    display(aoi_analysis_df.loc[is_no_dwell])
    display(aoi_analysis_df.loc[~is_no_dwell])
    # aoi_analysis_df = aoi_analysis_df[~is_no_dwell]
    plot_dwell_time_distributions(aoi_analysis_df, SELF_LIE, to_file=self_lie_distribution_plot)
    plot_dwell_time_distributions(aoi_analysis_df, SELF_TRUE, to_file=self_true_distribution_plot)
    plot_dwell_time_distributions(aoi_analysis_df, OTHER_LIE, to_file=other_lie_distribution_plot)
    plot_dwell_time_distributions(aoi_analysis_df, OTHER_TRUTH, to_file=other_true_distribution_plot)


def pid_dtw_analysis(input_distance_file: str = DTW_T_CSV,
                     input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                     pid_matrix_plot: str = PID_DISTANCE_PLOT,
                     pid_percent_lies_plot: str = PID_PERCENT_LIES_PLOT,
                     pid_dwell_times_plot: str = PID_DWELL_TIMES_PLOT,
                     pid_n_transitions_plot: str = PID_N_TRANSITIONS_PLOT,
                     percent_lies_by_pid_plot: str = PERCENT_LIES_BY_PID_PLOT,
                     max_clusters: int = 20,
                     colors: List[str] = XKCD_COLORS_LIST):

    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    big_matrix_df = create_big_matrix(distance_df)
    matrix_df = create_pid_matrix(big_matrix_df, fill_self_distances=0)
    cluster_df = get_best_fit_heirarchical_clusters(matrix_df, get_heirarchical_clusters_by_pid, max_clusters=max_clusters)
    plot_pid_matrix_with_clusters(matrix_df, cluster_df, colors=colors, to_file=pid_matrix_plot)
    responses_df = get_pid_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, PID, colors, to_file=pid_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, PID, colors, to_file=pid_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, PID, colors, to_file=pid_n_transitions_plot)
    responses_df_by_pid = get_response_stats_by_pid(cluster_df, aoi_analysis_df)
    plot_percent_lies_by_pid(responses_df_by_pid, colors, to_file=percent_lies_by_pid_plot)


def trial_id_dtw_analysis(input_distance_file: str = DTW_T_CSV,
                          input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                          trial_id_matrix_plot: str = TRIAL_DISTANCE_PLOT,
                          trial_id_percent_lies_plot: str = TRIAL_PERCENT_LIES_PLOT,
                          trial_id_dwell_times_plot: str = TRIAL_DWELL_TIMES_PLOT,
                          trial_id_n_transitions_plot: str = TRIAL_N_TRANSITIONS_PLOT,
                          percent_lies_by_trial_id_plot: str = PERCENT_LIES_BY_TRIAL_ID_PLOT,
                          max_clusters: int = 20,
                          colors: List[str] = XKCD_COLORS_LIST):
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    big_matrix_df = create_big_matrix(distance_df)
    matrix_df = create_trial_id_matrix(big_matrix_df, fill_self_distances=0)
    cluster_df = get_best_fit_heirarchical_clusters(matrix_df, get_heirarchical_clusters_by_trial_id, max_clusters=max_clusters)
    plot_trial_id_matrix_with_clusters(matrix_df, cluster_df, colors=colors, to_file=trial_id_matrix_plot)
    responses_df = get_trial_id_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL_ID, colors, to_file=trial_id_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL_ID, colors, to_file=trial_id_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL_ID, colors, to_file=trial_id_n_transitions_plot)
    responses_df_by_trial_id = get_response_stats_by_trial_id(cluster_df, aoi_analysis_df)
    plot_percent_lies_by_trial_id(responses_df_by_trial_id, colors, to_file=percent_lies_by_trial_id_plot)


def trial_count_dtw_analysis(input_distance_file: str = DTW_T_CSV,
                             input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                             trial_count_matrix_plot: str = TRIAL_COUNT_DISTANCE_PLOT,
                             trial_count_percent_lies_plot: str = TRIAL_COUNT_PERCENT_LIES_PLOT,
                             trial_count_dwell_times_plot: str = TRIAL_COUNT_DWELL_TIMES_PLOT,
                             trial_count_n_transitions_plot: str = TRIAL_COUNT_N_TRANSITIONS_PLOT,
                             max_clusters: int = 20,
                             colors: List[str] = XKCD_COLORS_LIST):
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    big_matrix_df = create_big_matrix(distance_df)
    matrix_df = create_trial_count_matrix(big_matrix_df, fill_self_distances=0)
    cluster_df = get_best_fit_heirarchical_clusters(matrix_df, get_heirarchical_clusters_by_trial_count, max_clusters=max_clusters)
    plot_trial_count_matrix_with_clusters(matrix_df, cluster_df, colors=colors, to_file=trial_count_matrix_plot)
    responses_df = get_trial_count_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL_COUNT, colors, to_file=trial_count_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL_COUNT, colors, to_file=trial_count_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL_COUNT, colors, to_file=trial_count_n_transitions_plot)


def all_trial_dtw_analysis(input_distance_file: str = DTW_T_CSV,
                           input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                           all_trial_percent_lies_plot: str = ALL_TRIAL_PERCENT_LIES_PLOT,
                           all_trial_dwell_times_plot: str = ALL_TRIAL_DWELL_TIMES_PLOT,
                           all_trial_n_transitions_plot: str = ALL_TRIAL_N_TRANSITIONS_PLOT,
                           max_clusters: int = 20,
                           colors: List[str] = XKCD_COLORS_LIST):
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    big_matrix_df = create_big_matrix(distance_df)
    big_matrix_df.columns = big_matrix_df.columns.droplevel()
    big_matrix_df = set_diagonal(big_matrix_df)
    cluster_df = get_best_fit_heirarchical_clusters(big_matrix_df, get_heirarchical_clusters_by_trial, max_clusters=max_clusters)
    aoi_analysis_df.set_index([aoi_analysis_df.index, TRIAL_COUNT], inplace=True)
    responses_df = get_trial_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL, colors, to_file=all_trial_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL, colors, to_file=all_trial_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL, colors, to_file=all_trial_n_transitions_plot)


def kmeans_analysis(input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                    percent_lies_plot: str = KMEANS_PERCENT_LIES_PLOT,
                    dwell_times_plot: str = KMEANS_DWELL_TIMES_PLOT,
                    n_transitions_plot: str = KMEANS_N_TRANSITIONS_PLOT,
                    columns: List[str] = [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH, PAYNE_INDEX],
                    max_clusters: int = 20,
                    colors: List[str] = XKCD_COLORS_LIST[20:]):

    title_prefix = 'KMeans: '

    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    aoi_analysis_df = aoi_analysis_df[~is_no_dwell_for_aois(aoi_analysis_df)]
    for_kmeans_df = prepare_data(aoi_analysis_df, columns)
    cluster_df = get_best_fit_kmeans_clusters(for_kmeans_df, n_components=None, max_clusters=max_clusters)
    display(cluster_df)
    responses_df = get_trial_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_transitions_plot)


def proximal_analysis(input_distance_file: str = DTW_T_CSV,
                      input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                      window_size: int = 7):
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    big_matrix_df = create_big_matrix(distance_df)
    proximal_df, distal_df = get_proximal_and_distal_distances(big_matrix_df, window_size=window_size)
    plot_distance_distribution(proximal_df, 'for Proximal Pairs')
    plot_distance_distribution(distal_df, 'for Distal Pairs')
