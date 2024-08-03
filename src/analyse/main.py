from typing import List
from pandas import DataFrame
from analyse.dtw_analysis import create_big_matrix, create_pid_matrix, create_trial_count_matrix, create_trial_id_matrix, get_dbscan_clusters, get_heirarchical_clusters_by_pid, get_heirarchical_clusters_by_trial, get_heirarchical_clusters_by_trial_count, get_heirarchical_clusters_by_trial_id, get_kmedoids_clusters, kmedoids, plot_distance_distribution, plot_nearest_neighbour_points, plot_pid_matrix_with_clusters, plot_trial_count_matrix_with_clusters, plot_trial_id_matrix_with_clusters, get_proximal_and_distal_distances, set_diagonal
from analyse.get_response_stats import get_gains_response_stats
from analyse.kmeans_analysis import cluster_analysis, get_kmeans_clusters, merge_components, plot_correlation_matrix, prepare_data
from analyse.n_cluster_analysis import get_best_fit_heirarchical_clusters, get_best_fit_partitional_clusters, get_best_fit_partitional_clusters_from_features, get_best_fit_partitional_clusters_from_matrix
from analyse.response_analysis import XKCD_COLORS_LIST, calculate_mean_response_stat, do_anova, do_clustered_pid_t_test, do_gains_t_test, do_lie_percentage_chi_squared_tests, do_clustered_by_pid_mean_t_test, get_confidence_intervals, get_pid_response_stats_for_clusters_by_gain_label, get_pid_response_stats_no_clusters, get_trial_id_response_stats_no_clusters, get_trial_response_stats_by_pid, get_trial_response_stats_no_clusters, get_trials_by_cluster, plot_aoi_by_pid, plot_dwell_time_by_gain_for_clusters, plot_avg_dwell_time_distributions, get_pid_response_stats_for_clusters, get_response_stats_by_pid, get_response_stats_by_trial_id, get_trial_count_response_stats_for_clusters, get_trial_id_response_stats_for_clusters, get_trial_response_stats_for_clusters, plot_dwell_time_distribution, plot_dwell_times_for_clusters, plot_gain_of_ten_by_pid, plot_gain_for_clusters, plot_gain_under_ten_by_pid, plot_gains_avg_dwell_time, plot_gains_by_trial_id, plot_gains_mean_percent_lie, plot_gains_n_transitions, plot_losses_avg_dwell_time, plot_losses_n_transitions, plot_n_transitions_by_gain_for_clusters, plot_n_transitions_by_pid, plot_n_transitions_distributions, plot_n_transitions_for_clusters, plot_n_trials_for_clusters, plot_n_trials_for_clusters_by_pid, plot_percent_lie_by_gain_for_clusters, plot_percent_lies_by_pid, plot_percent_lies_by_trial_id, plot_percent_lies_for_clusters, plot_response_stats_for_clusters, plot_rt_distributions, simple_plot, sort_response_df_by_pid_lie_percent, sort_response_df_by_pid_lie_percent_in_cluster
from preprocess.trial_id import calculate_gains_losses
from utils.columns import AVG_DWELL, CLUSTER, DISTANCE, GAIN_OF_TEN, GAIN_OF_THIRTY, GAIN_OF_TWENTY, GAIN_UNDER_TEN, GAIN_UNDER_THIRTY, LIE, LOSS_OF_TEN, LOSS_OF_THIRTY, LOSS_OF_TWENTY, LOSS_UNDER_TEN, LOSS_UNDER_THIRTY, N_ALT_TRANSITIONS, N_ATT_TRANSITIONS, N_TRANSITIONS, NEGATIVE_GAIN, OTHER_LIE, OTHER_LOSS, OTHER_TRUTH, PAYNE_INDEX, PID, POSITIVE_GAIN, RT, SELF_GAIN, SELF_LIE, SELF_TRUE, TRIAL, TRIAL_COUNT, TRIAL_ID, UNIQUE_AOIS
from utils.display import display
from utils.masks import get_gain_of_less_than_ten, get_gain_of_ten, get_loss_of_less_than_ten, get_loss_of_ten, is_no_dwell_for_aois
from utils.paths import *
from utils.read_csv import read_friom_cluster_files_to_dict, read_from_analysis_file, read_from_cluster_file, read_from_dtw_file, read_from_dwell_file, read_from_trial_index_file
from sklearn.preprocessing import StandardScaler, RobustScaler
from matplotlib.colors import XKCD_COLORS
from contextlib import contextmanager

COLORS = XKCD_COLORS_LIST


def descriptives_analysis(input_aoi_analysis_file: str = None):
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]] = aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]] * 1000
    trial_id_response_df = get_trial_id_response_stats_no_clusters(aoi_analysis_df, with_error_bounds=False)
    display(calculate_mean_response_stat(trial_id_response_df, LIE, with_error_bounds=False))
    display(get_confidence_intervals(trial_id_response_df[LIE]))
    pid_response_df = get_pid_response_stats_no_clusters(aoi_analysis_df, with_error_bounds=False)
    display(calculate_mean_response_stat(pid_response_df, LIE, with_error_bounds=False))
    display(get_confidence_intervals(trial_id_response_df[LIE]))
    trial_response_df = get_trial_response_stats_no_clusters(aoi_analysis_df, with_error_bounds=False, val_only=True)
    display(trial_response_df)
    display(aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]].mean())
    display(aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]].std())
    display(aoi_analysis_df[SELF_LIE].value_counts())
    display(aoi_analysis_df[SELF_TRUE].value_counts())
    display(aoi_analysis_df[OTHER_LIE].value_counts())
    display(aoi_analysis_df[OTHER_TRUTH].value_counts())
    gains_response_df = get_gains_response_stats(aoi_analysis_df, [NEGATIVE_GAIN, GAIN_UNDER_TEN, GAIN_OF_TEN, GAIN_OF_TWENTY, GAIN_OF_THIRTY], with_error_bounds=False)
    display(gains_response_df)
    gains_response_df = get_gains_response_stats(aoi_analysis_df, [LOSS_UNDER_TEN, LOSS_OF_TEN, LOSS_OF_TWENTY, LOSS_OF_THIRTY], with_error_bounds=False)
    gains_response_df = get_gains_response_stats(aoi_analysis_df, [f"{LOSS_OF_THIRTY}, {GAIN_OF_TEN}", f"{LOSS_OF_THIRTY}, {GAIN_UNDER_TEN}"], with_error_bounds=False)
    display(gains_response_df)
    # trial_id_response_df = get_trial_id_response_stats_no_clusters(aoi_analysis_df, with_error_bounds=False, val_only=True)
    # display(DataFrame([trial_id_response_df.min(), trial_id_response_df.max()]))
    # pid_response_df = get_pid_response_stats_no_clusters(aoi_analysis_df, with_error_bounds=False, val_only=True)
    # display(pid_response_df)
    # display(DataFrame([pid_response_df.min(), pid_response_df.max()]))
    # display(trial_response_df)


def response_analysis(input_aoi_analysis_file: str = None,
                      input_trial_index_file: str = None,
                      n_trials_by_pid_plot: str = None,
                      percent_lies_by_pid_plot: str = None,
                      percent_lies_by_trial_id_plot: str = None,
                      net_gain_lie_plot: str = None,
                      net_loss_lie_plot: str = None,
                      net_gain_loss_lie_plot: str = None,
                      avg_dwell_per_gain_plot: str = None,
                      avg_n_transition_per_gain_plot: str = None,
                      avg_dwell_per_loss_plot: str = None,
                      avg_n_transition_per_loss_plot: str = None,
                      output_trial_index_gains_plot: str = None,
                      colors: List[str] = XKCD_COLORS_LIST):
    purple = XKCD_COLORS["xkcd:purple"]
    med_blue, blue = XKCD_COLORS["xkcd:light blue"], XKCD_COLORS["xkcd:medium blue"]
    green, med_green = XKCD_COLORS["xkcd:medium green"], XKCD_COLORS["xkcd:greenish"]

    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    response_df = get_pid_response_stats_no_clusters(aoi_analysis_df)
    sorted_responses_by_pid_df = sort_response_df_by_pid_lie_percent(response_df, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_responses_by_pid_df, colors=[purple], to_file=percent_lies_by_pid_plot)
    plot_n_trials_for_clusters_by_pid(sorted_responses_by_pid_df, PID, [XKCD_COLORS["xkcd:crimson"]], title_override="", to_file=n_trials_by_pid_plot)
    trial_id_response_df = get_trial_id_response_stats_no_clusters(aoi_analysis_df)
    plot_percent_lies_by_trial_id(trial_id_response_df.sort_values(LIE), colors=[purple], to_file=percent_lies_by_trial_id_plot)
    trial_index_df = read_from_trial_index_file(input_trial_index_file)
    gains_df = calculate_gains_losses(trial_index_df)
    plot_gains_by_trial_id(gains_df, to_file=output_trial_index_gains_plot)
    display(gains_df.median())
    display(gains_df.quantile(0.25))
    display(gains_df.quantile(0.75))
    display(gains_df.mean())
    display(gains_df.std())
    title = ""
    # title = "Percent of Lie based on Net Gain to Sender from lying in Trial"
    plot_gains_mean_percent_lie(aoi_analysis_df, [GAIN_UNDER_TEN, GAIN_OF_TEN, GAIN_OF_TWENTY, GAIN_OF_THIRTY],
                                title=title, colors=[XKCD_COLORS["xkcd:pinkish purple"], XKCD_COLORS["xkcd:purply"], purple, XKCD_COLORS["xkcd:darkish purple"]], to_file=net_gain_lie_plot)
    # title = "Percent of Lie based on Net Loss to Receiver from lying in Trial"
    plot_gains_mean_percent_lie(aoi_analysis_df, [LOSS_UNDER_TEN, LOSS_OF_TEN, LOSS_OF_TWENTY, LOSS_OF_THIRTY],
                                title=title, colors=[med_blue, blue, XKCD_COLORS["xkcd:dusk blue"], XKCD_COLORS["xkcd:darkish blue"]], to_file=net_loss_lie_plot)
    plot_gains_avg_dwell_time(aoi_analysis_df, [GAIN_UNDER_TEN, GAIN_OF_TEN], [XKCD_COLORS["xkcd:pinkish purple"], XKCD_COLORS["xkcd:purply"]], to_file=avg_dwell_per_gain_plot)
    plot_gains_avg_dwell_time(aoi_analysis_df, [GAIN_UNDER_THIRTY, GAIN_OF_THIRTY], [purple, XKCD_COLORS["xkcd:darkish purple"]], to_file=avg_dwell_per_gain_plot)
    plot_gains_n_transitions(aoi_analysis_df, [GAIN_UNDER_TEN, GAIN_OF_TEN], [purple, XKCD_COLORS["xkcd:darkish purple"]], to_file=avg_n_transition_per_gain_plot)
    plot_gains_n_transitions(aoi_analysis_df, [GAIN_UNDER_THIRTY, GAIN_OF_THIRTY], [purple, XKCD_COLORS["xkcd:darkish purple"]], to_file=avg_dwell_per_gain_plot)
    # plot_gains_n_transitions(aoi_analysis_df, [NEGATIVE_GAIN, POSITIVE_GAIN], [sand, purple], to_file=avg_n_transition_per_gain_plot)
    plot_losses_avg_dwell_time(aoi_analysis_df, [LOSS_UNDER_TEN, LOSS_OF_TEN], [med_blue, blue], to_file=avg_dwell_per_loss_plot)
    plot_losses_n_transitions(aoi_analysis_df, [LOSS_UNDER_TEN, LOSS_OF_TEN], [med_blue, blue], to_file=avg_n_transition_per_loss_plot)

    for_test = [(GAIN_OF_TEN, GAIN_UNDER_TEN), (LOSS_OF_TEN, LOSS_UNDER_TEN), (GAIN_UNDER_THIRTY, GAIN_OF_THIRTY), (LOSS_UNDER_THIRTY, LOSS_OF_THIRTY)]
    do_lie_percentage_chi_squared_tests(aoi_analysis_df, for_test)
    for_test = [(GAIN_OF_TEN, GAIN_UNDER_TEN), (LOSS_OF_TEN, LOSS_UNDER_TEN), (GAIN_UNDER_THIRTY, GAIN_OF_THIRTY)]
    do_gains_t_test(aoi_analysis_df, for_test).pipe(display)


def pid_dtw_analysis(input_distance_file: str = None,
                     input_aoi_analysis_file: str = None,
                     pid_matrix_plot: str = None,
                     pid_percent_lies_plot: str = None,
                     pid_dwell_times_plot: str = None,
                     pid_n_transitions_plot: str = None,
                     percent_lies_by_pid_plot: str = None,
                     percent_lies_gain_cluster_plot: str = None,
                     n_transitions_gain_cluster_plot: str = None,
                     self_lie_gain_cluster_plot: str = None,
                     self_true_gain_cluster_plot: str = None,
                     other_lie_gain_cluster_plot: str = None,
                     other_truth_gain_cluster_plot: str = None,
                     max_clusters: int = 20,
                     n_clusters: int = None,
                     colors: List[str] = XKCD_COLORS_LIST):
    display(input_distance_file)
    distance_df = read_from_dtw_file(input_distance_file)
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file, [PID, TRIAL_ID, TRIAL_COUNT])
    big_matrix_df = create_big_matrix(distance_df, filter_by_df=aoi_analysis_df)
    matrix_df = create_pid_matrix(big_matrix_df, fill_self_distances=None)
    cluster_df = get_best_fit_heirarchical_clusters(matrix_df, get_heirarchical_clusters_by_pid,
                                                    max_clusters=max_clusters) if not n_clusters else get_heirarchical_clusters_by_pid(matrix_df, n_clusters)
    plot_pid_matrix_with_clusters(matrix_df, cluster_df, colors=colors, to_file=pid_matrix_plot)
    responses_df = get_pid_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    display(responses_df)
    plot_percent_lies_for_clusters(responses_df, PID, colors, to_file=pid_percent_lies_plot)
    plot_dwell_times_for_clusters(responses_df, PID, colors, to_file=pid_dwell_times_plot)
    plot_n_transitions_for_clusters(responses_df, PID, colors, to_file=pid_n_transitions_plot)
    responses_df_by_pid = get_response_stats_by_pid(cluster_df, aoi_analysis_df)
    sorted_df = sort_response_df_by_pid_lie_percent_in_cluster(responses_df_by_pid, aoi_analysis_df)
    plot_percent_lies_by_pid(sorted_df, colors, to_file=percent_lies_by_pid_plot)
    plot_aoi_by_pid(sorted_df, SELF_LIE, colors)
    plot_aoi_by_pid(sorted_df, SELF_TRUE, colors)
    plot_aoi_by_pid(sorted_df, OTHER_LIE, colors)
    plot_aoi_by_pid(sorted_df, OTHER_TRUTH, colors)
    plot_n_transitions_by_pid(sorted_df, colors)

    gain_under_ten_responses_df = get_pid_response_stats_for_clusters_by_gain_label(cluster_df, GAIN_UNDER_TEN, aoi_analysis_df)
    display(get_pid_response_stats_for_clusters_by_gain_label(cluster_df, GAIN_UNDER_TEN, aoi_analysis_df, with_error_bounds=False), max_cols=None, title=GAIN_UNDER_TEN)
    gain_of_ten_responses_df = get_pid_response_stats_for_clusters_by_gain_label(cluster_df, GAIN_OF_TEN, aoi_analysis_df)
    display(get_pid_response_stats_for_clusters_by_gain_label(cluster_df, GAIN_OF_TEN, aoi_analysis_df, with_error_bounds=False), max_cols=None, title=GAIN_OF_TEN)

    plot_percent_lie_by_gain_for_clusters(gain_under_ten_responses_df, gain_of_ten_responses_df, colors, to_file=percent_lies_gain_cluster_plot)
    plot_n_transitions_by_gain_for_clusters(gain_under_ten_responses_df, gain_of_ten_responses_df, colors, to_file=n_transitions_gain_cluster_plot)
    plot_dwell_time_by_gain_for_clusters(gain_under_ten_responses_df, gain_of_ten_responses_df, SELF_LIE, colors, to_file=self_lie_gain_cluster_plot)
    plot_dwell_time_by_gain_for_clusters(gain_under_ten_responses_df, gain_of_ten_responses_df, SELF_TRUE, colors, to_file=self_true_gain_cluster_plot)
    plot_dwell_time_by_gain_for_clusters(gain_under_ten_responses_df, gain_of_ten_responses_df, OTHER_LIE, colors, to_file=other_lie_gain_cluster_plot)
    plot_dwell_time_by_gain_for_clusters(gain_under_ten_responses_df, gain_of_ten_responses_df, OTHER_TRUTH, colors, to_file=other_truth_gain_cluster_plot)

    # responses_df = get_pid_response_stats_for_clusters(cluster_df, aoi_analysis_df, with_error_bounds=False)
    # display(responses_df, max_cols=None)
    # responses_df_by_pid = get_response_stats_by_pid(cluster_df, aoi_analysis_df, with_error_bounds=False)
    # display(responses_df_by_pid[CLUSTER].value_counts())
    # display(calculate_mean_response_stat(responses_df_by_pid[responses_df_by_pid[CLUSTER] == 1], LIE, with_error_bounds=False))
    # display(calculate_mean_response_stat(responses_df_by_pid[responses_df_by_pid[CLUSTER] == 2], LIE, with_error_bounds=False))

    display(cluster_df.value_counts())
    display(do_clustered_pid_t_test(aoi_analysis_df, cluster_df), max_cols=None)
    display(do_clustered_by_pid_mean_t_test(aoi_analysis_df, cluster_df, [LIE, SELF_TRUE, N_TRANSITIONS]))
    # do_anova(aoi_analysis_df, cluster_df)


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
    display(responses_by_pid_df)
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


def distribution_analysis(input_aoi_analysis_file: str = None,
                          input_dwell_timeline_file: str = None,
                          dwell_distribution_plot: str = None,
                          rt_distribution_plot: str = None,
                          n_transition_distribution_plot: str = None):
    aoi_analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]] = aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]] * 1000
    dwell_df = read_from_dwell_file(input_dwell_timeline_file)
    # is_no_dwell = is_no_dwell_for_aois(aoi_analysis_df, [SELF_TRUE, SELF_LIE, OTHER_TRUTH, OTHER_LIE])
    # display(aoi_analysis_df.loc[is_no_dwell])
    # display(aoi_analysis_df.loc[~is_no_dwell])
    plot_dwell_time_distribution(dwell_df, to_file=None)
    plot_rt_distributions(aoi_analysis_df, to_file=rt_distribution_plot)
    plot_avg_dwell_time_distributions(aoi_analysis_df, to_file=dwell_distribution_plot)
    plot_n_transitions_distributions(aoi_analysis_df, to_file=n_transition_distribution_plot)


class SaveToFileFn:
    def __init__(self, save):
        self.save = save

    def __enter__(self):
        return lambda f: f(self.save)

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


def do_analyses(do_save: bool = False, **params):
    with SaveToFileFn(do_save) as save:
        # pid_dtw_analysis(**save(params['pid_dtw']))
        # response_analysis(**save(params["response"]))
        # distribution_analysis(**save(params["distribution"]))
        # proximal_analysis(**save(params["proximal"]))
        # trial_id_dtw_analysis(**save(params["trial_id_dtw"]))
        # descriptives_analysis(**save(params["descriptives"]))
        kmedoids_dtw_analysis(**save(params["kmedoids"]))
