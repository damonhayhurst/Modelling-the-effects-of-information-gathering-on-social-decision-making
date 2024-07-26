from typing import List
from pandas import DataFrame
from analyse.dtw_analysis import create_big_matrix, create_pid_matrix, create_trial_count_matrix, create_trial_id_matrix, get_dbscan_clusters, get_heirarchical_clusters_by_pid, get_heirarchical_clusters_by_trial, get_heirarchical_clusters_by_trial_count, get_heirarchical_clusters_by_trial_id, get_kmedoids_clusters, kmedoids, plot_distance_distribution, plot_nearest_neighbour_points, plot_pid_matrix_with_clusters, plot_trial_count_matrix_with_clusters, plot_trial_id_matrix_with_clusters, get_proximal_and_distal_distances, set_diagonal
from analyse.kmeans_analysis import cluster_analysis, get_kmeans_clusters, merge_components, plot_correlation_matrix, prepare_data
from analyse.n_cluster_analysis import get_best_fit_heirarchical_clusters, get_best_fit_partitional_clusters, get_best_fit_partitional_clusters_from_features, get_best_fit_partitional_clusters_from_matrix
from analyse.response_analysis import XKCD_COLORS_LIST, calculate_mean_response_stat, do_clustered_pid_t_test, do_gains_t_test, get_pid_response_stats_for_clusters_by_gain_label, get_pid_response_stats_no_clusters, get_trial_id_response_stats_no_clusters, get_trial_response_stats_by_pid, get_trials_by_cluster, plot_dwell_time_distributions, get_pid_response_stats_for_clusters, get_response_stats_by_pid, get_response_stats_by_trial_id, get_trial_count_response_stats_for_clusters, get_trial_id_response_stats_for_clusters, get_trial_response_stats_for_clusters, plot_dwell_times_for_clusters, plot_gain_of_ten_by_pid, plot_gain_for_clusters, plot_gain_under_ten_by_pid, plot_gains_avg_dwell_time, plot_gains_mean_percent_lie, plot_gains_n_transitions, plot_losses_avg_dwell_time, plot_losses_mean_percent_lie, plot_losses_n_transitions, plot_n_transitions_for_clusters, plot_n_trials_for_clusters, plot_n_trials_for_clusters_by_pid, plot_percent_lies_by_pid, plot_percent_lies_by_trial_id, plot_percent_lies_for_clusters, plot_response_stats_for_clusters, plot_rt_distributions, simple_plot, sort_response_df_by_pid_lie_percent, sort_response_df_by_pid_lie_percent_in_cluster
from preprocess.trial_id import calculate_gains_losses
from utils.columns import AVG_DWELL, CLUSTER, DISTANCE, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_OF_TWENTY, GAIN_UNDER_TEN, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_OF_TWENTY, LOSS_UNDER_TEN, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, NEGATIVE_GAIN, OTHER_LIE, OTHER_LOSS, OTHER_TRUTH, PAYNE_INDEX, PID, POSITIVE_GAIN, RT, SELF_GAIN, SELF_LIE, SELF_TRUE, TRIAL, TRIAL_COUNT, TRIAL_ID, UNIQUE_AOIS
from utils.display import display
from utils.masks import get_gain_of_ten, is_no_dwell_for_aois
from utils.paths import *
from utils.read_csv import read_friom_cluster_files_to_dict, read_from_analysis_file, read_from_cluster_file, read_from_dtw_file, read_from_dwell_file, read_from_trial_index_file
from sklearn.preprocessing import StandardScaler, RobustScaler
from matplotlib.colors import XKCD_COLORS

COLORS = XKCD_COLORS_LIST


def response_analysis(input_aoi_analysis_file: str = None,
                      input_trial_index_file: str = None,
                      n_trials_by_pid_plot: str = None,
                      percent_lies_by_pid_plot: str = None,
                      percent_lies_by_trial_id_plot: str = None,
                      net_gain_lie_plot: str = None,
                      net_loss_lie_plot: str = None,
                      avg_dwell_per_gain_plot: str = None,
                      avg_n_transition_per_gain_plot: str = None,
                      avg_dwell_per_negative_gain_plot: str = None,
                      avg_dwell_per_loss_plot: str = None,
                      avg_n_transition_per_loss_plot: str = None,
                      colors: List[str] = XKCD_COLORS_LIST):
    med_purple, purple, sand = XKCD_COLORS["xkcd:pinkish purple"], XKCD_COLORS["xkcd:purple"], XKCD_COLORS["xkcd:light tan"]
    med_blue, blue = XKCD_COLORS["xkcd:light blue"], XKCD_COLORS["xkcd:medium blue"]

    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    response_df = get_pid_response_stats_no_clusters(aoi_analysis_df)
    sorted_responses_by_pid_df = sort_response_df_by_pid_lie_percent(response_df, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_responses_by_pid_df, colors, to_file=percent_lies_by_pid_plot)
    plot_n_trials_for_clusters_by_pid(sorted_responses_by_pid_df, PID, colors, to_file=n_trials_by_pid_plot)
    trial_id_response_df = get_trial_id_response_stats_no_clusters(aoi_analysis_df).sort_values(LIE)
    plot_percent_lies_by_trial_id(trial_id_response_df, to_file=percent_lies_by_trial_id_plot)
    trial_index_df = read_from_trial_index_file(input_trial_index_file)
    gains_df = calculate_gains_losses(trial_index_df)
    display(gains_df.median())
    plot_gains_mean_percent_lie(aoi_analysis_df, [NEGATIVE_GAIN, GAIN_UNDER_TEN, GAIN_OF_TEN, GAIN_OF_TWENTY, GAIN_OF_THIRTY], [sand, med_purple, purple, purple, purple], to_file=net_gain_lie_plot)
    plot_losses_mean_percent_lie(aoi_analysis_df, [LOSS_UNDER_TEN, LOSS_OF_TEN, LOSS_OF_TWENTY, LOSS_OF_THIRTY], [med_blue, blue, blue, blue], to_file=net_loss_lie_plot)
    plot_gains_avg_dwell_time(aoi_analysis_df, [GAIN_UNDER_TEN, GAIN_OF_TEN], [med_purple, purple], to_file=avg_dwell_per_gain_plot)
    plot_gains_avg_dwell_time(aoi_analysis_df, [NEGATIVE_GAIN, POSITIVE_GAIN], [sand, purple], to_file=avg_dwell_per_negative_gain_plot)
    plot_gains_n_transitions(aoi_analysis_df, [GAIN_UNDER_TEN, GAIN_OF_TEN], [med_purple, purple], to_file=avg_n_transition_per_gain_plot)
    plot_gains_n_transitions(aoi_analysis_df, [NEGATIVE_GAIN, POSITIVE_GAIN], [sand, purple], to_file=avg_n_transition_per_gain_plot)
    plot_losses_avg_dwell_time(aoi_analysis_df, [LOSS_UNDER_TEN, LOSS_OF_TEN], [med_blue, blue], to_file=avg_dwell_per_loss_plot)
    plot_losses_n_transitions(aoi_analysis_df, [LOSS_UNDER_TEN, LOSS_OF_TEN], [med_blue, blue], to_file=avg_n_transition_per_loss_plot)
    do_gains_t_test(aoi_analysis_df).pipe(display, max_cols=None)


def pid_dtw_analysis(input_distance_file: str = None,
                     input_aoi_analysis_file: str = None,
                     pid_matrix_plot: str = None,
                     pid_percent_lies_plot: str = None,
                     pid_dwell_times_plot: str = None,
                     pid_n_transitions_plot: str = None,
                     percent_lies_by_pid_plot: str = None,
                     max_clusters: int = 20,
                     n_clusters: int = None,
                     colors: List[str] = XKCD_COLORS_LIST):

    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file, [PID, TRIAL_ID, TRIAL_COUNT])
    big_matrix_df = create_big_matrix(distance_df, filter_by_df=aoi_analysis_df)
    matrix_df = create_pid_matrix(big_matrix_df, fill_self_distances=None)
    cluster_df = get_best_fit_heirarchical_clusters(matrix_df, get_heirarchical_clusters_by_pid,
                                                    max_clusters=max_clusters) if not n_clusters else get_heirarchical_clusters_by_pid(matrix_df, n_clusters)
    plot_pid_matrix_with_clusters(matrix_df, cluster_df, colors=colors, to_file=pid_matrix_plot)
    responses_df = get_pid_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, PID, colors, to_file=pid_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, PID, colors, to_file=pid_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, PID, colors, to_file=pid_n_transitions_plot)
    responses_df_by_pid = get_response_stats_by_pid(cluster_df, aoi_analysis_df)
    sorted_df = sort_response_df_by_pid_lie_percent_in_cluster(responses_df_by_pid, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_df, colors, to_file=percent_lies_by_pid_plot)
    gain_under_ten_responses_df = get_pid_response_stats_for_clusters_by_gain_label(cluster_df, GAIN_UNDER_TEN, aoi_analysis_df)
    gain_of_ten_responses_df = get_pid_response_stats_for_clusters_by_gain_label(cluster_df, GAIN_OF_TEN, aoi_analysis_df)
    plot_percent_lies_for_clusters(gain_under_ten_responses_df, PID, colors, title_prefix=f"{GAIN_UNDER_TEN}: ", to_file=None)
    plot_percent_lies_for_clusters(gain_of_ten_responses_df, PID, colors, title_prefix=f"{GAIN_OF_TEN}: ", to_file=None)
    plot_dwell_times_for_clusters(gain_under_ten_responses_df, PID, colors, title_prefix=f"{GAIN_UNDER_TEN}: ", to_file=None)
    plot_dwell_times_for_clusters(gain_of_ten_responses_df, PID, colors, title_prefix=f"{GAIN_OF_TEN}: ", to_file=None)
    plot_n_transitions_for_clusters(gain_under_ten_responses_df, PID, colors, title_prefix=f"{GAIN_UNDER_TEN}: ", to_file=None)
    plot_n_transitions_for_clusters(gain_of_ten_responses_df, PID, colors, title_prefix=f"{GAIN_OF_TEN}: ", to_file=None)
    do_clustered_pid_t_test(aoi_analysis_df, cluster_df)
    


def trial_id_dtw_analysis(input_distance_file: str = None,
                          input_aoi_analysis_file: str = None,
                          trial_id_matrix_plot: str = None,
                          trial_id_percent_lies_plot: str = None,
                          trial_id_dwell_times_plot: str = None,
                          trial_id_n_transitions_plot: str = None,
                          percent_lies_by_trial_id_plot: str = None,
                          max_clusters: int = 20,
                          colors: List[str] = XKCD_COLORS_LIST):
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    big_matrix_df = create_big_matrix(distance_df)
    matrix_df = create_trial_id_matrix(big_matrix_df, fill_self_distances=None)
    cluster_df = get_best_fit_heirarchical_clusters(matrix_df, get_heirarchical_clusters_by_trial_id, max_clusters=max_clusters)
    plot_trial_id_matrix_with_clusters(matrix_df, cluster_df, colors=colors, to_file=trial_id_matrix_plot)
    responses_df = get_trial_id_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL_ID, colors, to_file=trial_id_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL_ID, colors, to_file=trial_id_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL_ID, colors, to_file=trial_id_n_transitions_plot)
    responses_df_by_trial_id = get_response_stats_by_trial_id(cluster_df, aoi_analysis_df)
    plot_percent_lies_by_trial_id(responses_df_by_trial_id, colors, to_file=percent_lies_by_trial_id_plot)


def trial_count_dtw_analysis(input_distance_file: str = None,
                             input_aoi_analysis_file: str = None,
                             trial_count_matrix_plot: str = None,
                             trial_count_percent_lies_plot: str = None,
                             trial_count_dwell_times_plot: str = None,
                             trial_count_n_transitions_plot: str = None,
                             max_clusters: int = 20,
                             colors: List[str] = XKCD_COLORS_LIST):
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    big_matrix_df = create_big_matrix(distance_df)
    matrix_df = create_trial_count_matrix(big_matrix_df, fill_self_distances=None)
    cluster_df = get_best_fit_heirarchical_clusters(matrix_df, get_heirarchical_clusters_by_trial_count, max_clusters=max_clusters)
    plot_trial_count_matrix_with_clusters(matrix_df, cluster_df, colors=colors, to_file=trial_count_matrix_plot)
    responses_df = get_trial_count_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL_COUNT, colors, to_file=trial_count_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL_COUNT, colors, to_file=trial_count_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL_COUNT, colors, to_file=trial_count_n_transitions_plot)


def all_trial_dtw_analysis(input_distance_file: str = None,
                           input_aoi_analysis_file: str = None,
                           all_trial_percent_lies_plot: str = None,
                           all_trial_dwell_times_plot: str = None,
                           all_trial_n_transitions_plot: str = None,
                           all_trial_n_trials_plot: str = None,
                           all_trial_percent_lie_by_pid_plot: str = None,
                           all_trial_n_trials_by_pid_plot: str = None,
                           all_trial_gain_of_ten_by_pid_plot: str = None,
                           all_trial_gain_under_ten_by_pid_plot: str = None,
                           max_clusters: int = 20,
                           n_clusters: int = None,
                           colors: List[str] = XKCD_COLORS_LIST):
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file, index=[PID, TRIAL_ID, TRIAL_COUNT])
    big_matrix_df = create_big_matrix(distance_df, filter_by_df=aoi_analysis_df)
    cluster_df = get_best_fit_heirarchical_clusters(big_matrix_df, get_heirarchical_clusters_by_trial,
                                                    max_clusters=max_clusters) if not n_clusters else get_heirarchical_clusters_by_trial(big_matrix_df, n_clusters)
    responses_df = get_trial_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL, colors, to_file=all_trial_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL, colors, to_file=all_trial_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL, colors, to_file=all_trial_n_transitions_plot)
    plot_n_trials_for_clusters(responses_df, TRIAL, colors, to_file=all_trial_n_trials_plot)
    responses_by_pid_df = get_trial_response_stats_by_pid(cluster_df, aoi_analysis_df)
    sorted_responses_by_pid_df = sort_response_df_by_pid_lie_percent(responses_by_pid_df, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_responses_by_pid_df, colors, to_file=all_trial_percent_lie_by_pid_plot)
    plot_n_trials_for_clusters_by_pid(sorted_responses_by_pid_df, TRIAL, colors, to_file=all_trial_n_trials_by_pid_plot)
    plot_gain_of_ten_by_pid(sorted_responses_by_pid_df, colors, to_file=all_trial_gain_of_ten_by_pid_plot)
    plot_gain_under_ten_by_pid(sorted_responses_by_pid_df, colors, to_file=all_trial_gain_under_ten_by_pid_plot)


def kmeans_analysis(input_aoi_analysis_file: str = None,
                    percent_lies_plot: str = None,
                    dwell_times_plot: str = None,
                    n_transitions_plot: str = None,
                    n_trials_plot: str = None,
                    correlations_plot: str = None,
                    percent_lies_by_pid_plot: str = None,
                    n_trials_by_pid_plot: str = None,
                    gain_of_ten_by_pid_plot: str = None,
                    gain_under_ten_by_pid_plot: str = None,
                    columns: List[str] = [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH, PAYNE_INDEX],
                    max_clusters: int = 20,
                    colors: List[str] = XKCD_COLORS_LIST[22:]):

    title_prefix = 'KMeans: %s\n' % ", ".join(columns)

    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    for_kmeans_df = prepare_data(aoi_analysis_df, columns, scaler=RobustScaler)
    plot_correlation_matrix(for_kmeans_df, to_file=correlations_plot)
    cluster_df = get_best_fit_partitional_clusters_from_features(for_kmeans_df, get_kmeans_clusters, max_clusters)
    # display(get_trials_by_cluster(7, cluster_df, aoi_analysis_df))
    responses_df = get_trial_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_n_trials_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_trials_plot)
    plot_percent_lies_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_transitions_plot)
    plot_gain_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=None)
    responses_by_pid_df = get_trial_response_stats_by_pid(cluster_df, aoi_analysis_df)
    sorted_responses_by_pid_df = sort_response_df_by_pid_lie_percent(responses_by_pid_df, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=percent_lies_by_pid_plot)
    plot_n_trials_for_clusters_by_pid(sorted_responses_by_pid_df, TRIAL, colors, title_prefix, to_file=n_trials_by_pid_plot)
    plot_gain_of_ten_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=gain_of_ten_by_pid_plot)
    plot_gain_under_ten_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=gain_under_ten_by_pid_plot)


def time_series_kmeans_analysis(input_cluster_files: List[str] = None,
                                input_aoi_analysis_file: str = None,
                                input_distance_file: str = None,
                                percent_lies_plot: str = None,
                                dwell_times_plot: str = None,
                                n_transitions_plot: str = None,
                                n_trials_plot: str = None,
                                percent_lies_by_pid_plot: str = None,
                                n_trials_by_pid_plot: str = None,
                                gain_of_ten_by_pid_plot: str = None,
                                gain_under_ten_by_pid_plot: str = None,
                                colors: List[str] = XKCD_COLORS_LIST[24:]):

    title_prefix = 'TimeSeriesKMeans: '
    cluster_df_dict = read_friom_cluster_files_to_dict(input_cluster_files)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file, index=[PID, TRIAL_ID, TRIAL_COUNT])
    distance_df = read_from_dtw_file(input_distance_file)

    def get_ts_kmeans_clusters(_: DataFrame, n_clusters: int):
        return cluster_df_dict[n_clusters]
    display(cluster_df_dict)
    big_matrix_df = create_big_matrix(distance_df, filter_by_df=aoi_analysis_df)
    k_cluster = cluster_df_dict.keys()
    cluster_df = get_best_fit_partitional_clusters_from_matrix(big_matrix_df, get_ts_kmeans_clusters, k_cluster)
    responses_df = get_trial_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    responses_by_pid_df = get_trial_response_stats_by_pid(cluster_df, aoi_analysis_df)
    plot_n_trials_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_trials_plot)
    plot_percent_lies_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_transitions_plot)
    plot_gain_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=None)
    sorted_responses_by_pid_df = sort_response_df_by_pid_lie_percent(responses_by_pid_df, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=percent_lies_by_pid_plot)
    plot_n_trials_for_clusters_by_pid(sorted_responses_by_pid_df, TRIAL, colors, title_prefix, to_file=n_trials_by_pid_plot)
    plot_gain_of_ten_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=gain_of_ten_by_pid_plot)
    plot_gain_under_ten_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=gain_under_ten_by_pid_plot)


def proximal_analysis(input_distance_file: str = None,
                      window_size: int = 7):
    distance_df = read_from_dtw_file(input_distance_file)
    big_matrix_df = create_big_matrix(distance_df)
    proximal_df, distal_df = get_proximal_and_distal_distances(big_matrix_df, window_size=window_size)
    plot_distance_distribution(distance_df)
    plot_distance_distribution(proximal_df, 'for Proximal Pairs')
    plot_distance_distribution(distal_df, 'for Distal Pairs')


def kmedoids_dtw_analysis(input_distance_file: str = None,
                          input_aoi_analysis_file: str = None,
                          percent_lies_plot: str = None,
                          dwell_times_plot: str = None,
                          n_transitions_plot: str = None,
                          n_trials_plot: str = None,
                          percent_lies_by_pid_plot: str = None,
                          n_trials_by_pid_plot: str = None,
                          max_clusters: int = 20,
                          n_clusters: int = None,
                          colors: List[str] = XKCD_COLORS_LIST[30:]):

    title_prefix = 'KMedoids: '

    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file, index=[PID, TRIAL_ID, TRIAL_COUNT])
    big_matrix_df = create_big_matrix(distance_df, filter_by_df=aoi_analysis_df)
    cluster_df = get_best_fit_partitional_clusters_from_matrix(big_matrix_df, get_kmedoids_clusters, range(2, max_clusters + 1)) if not n_clusters else get_kmedoids_clusters(big_matrix_df, n_clusters)
    responses_df = get_trial_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_n_trials_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_trials_plot)
    plot_percent_lies_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_transitions_plot)
    responses_by_pid_df = get_trial_response_stats_by_pid(cluster_df, aoi_analysis_df)
    sorted_responses_by_pid_df = sort_response_df_by_pid_lie_percent(responses_by_pid_df, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=percent_lies_by_pid_plot)
    plot_n_trials_for_clusters_by_pid(sorted_responses_by_pid_df, TRIAL, colors, title_prefix, to_file=n_trials_by_pid_plot)


def dbscan_dtw_analysis(input_distance_file: str = None,
                        input_aoi_analysis_file: str = None,
                        percent_lies_plot: str = None,
                        dwell_times_plot: str = None,
                        n_transitions_plot: str = None,
                        n_trials_plot: str = None,
                        percent_lies_by_pid_plot: str = None,
                        n_trials_by_pid_plot: str = None,
                        eps: float = 3.27,
                        n_neightbours: int = 3,
                        colors: List[str] = XKCD_COLORS_LIST[40:]):

    title_prefix = 'DBScan: '

    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file, index=[PID, TRIAL_ID, TRIAL_COUNT])
    big_matrix_df = create_big_matrix(distance_df, filter_by_df=aoi_analysis_df)
    plot_nearest_neighbour_points(big_matrix_df)
    cluster_df = get_dbscan_clusters(big_matrix_df, eps=eps, n_neighbours=n_neightbours)
    responses_df = get_trial_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_n_trials_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_trials_plot)
    plot_percent_lies_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, TRIAL, colors, title_prefix, to_file=n_transitions_plot)
    responses_by_pid_df = get_trial_response_stats_by_pid(cluster_df, aoi_analysis_df)
    sorted_responses_by_pid_df = sort_response_df_by_pid_lie_percent(responses_by_pid_df, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_responses_by_pid_df, colors, title_prefix, to_file=percent_lies_by_pid_plot)
    plot_n_trials_for_clusters_by_pid(sorted_responses_by_pid_df, TRIAL, colors, title_prefix, to_file=n_trials_by_pid_plot)


def dwell_analysis(input_aoi_analysis_file: str = None,
                   self_lie_distribution_plot: str = None,
                   self_true_distribution_plot: str = None,
                   other_lie_distribution_plot: str = None,
                   other_true_distribution_plot: str = None,
                   rt_distribution_plot: str = None):
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    # is_no_dwell = is_no_dwell_for_aois(aoi_analysis_df, [SELF_TRUE, SELF_LIE, OTHER_TRUTH, OTHER_LIE])
    # display(aoi_analysis_df.loc[is_no_dwell])
    # display(aoi_analysis_df.loc[~is_no_dwell])
    plot_rt_distributions(aoi_analysis_df, to_file=rt_distribution_plot)
    plot_dwell_time_distributions(aoi_analysis_df, SELF_LIE, to_file=self_lie_distribution_plot)
    plot_dwell_time_distributions(aoi_analysis_df, SELF_TRUE, to_file=self_true_distribution_plot)
    plot_dwell_time_distributions(aoi_analysis_df, OTHER_LIE, to_file=other_lie_distribution_plot)
    plot_dwell_time_distributions(aoi_analysis_df, OTHER_TRUTH, to_file=other_true_distribution_plot)
