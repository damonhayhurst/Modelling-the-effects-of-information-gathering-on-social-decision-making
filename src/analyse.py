
from pandas import DataFrame

from analyse.dtw_analysis import XKCD_COLORS_LIST, create_big_matrix, create_pid_matrix, create_trial_id_matrix, get_heirarchical_clusters_by_pid, get_heirarchical_clusters_by_trial_id, plot_pid_matrix_with_clusters, plot_trial_id_matrix_with_clusters
from analyse.n_cluster_analysis import get_best_fit_clusters, n_cluster_analysis
from analyse.response_analysis import *
from utils.columns import PID, TRIAL_ID
from utils.display import display
from utils.read_csv import read_from_analysis_file
from utils.paths import DTW_PEN_Z_CSV, PID_DISTANCE_PLOT, PID_DWELL_TIMES_PLOT, PID_N_TRANSITIONS_PLOT, PID_PERCENT_LIES_PLOT, TRIAL_DISTANCE_PLOT, TRIAL_DWELL_TIMES_PLOT, TRIAL_N_TRANSITIONS_PLOT, TRIAL_PERCENT_LIES_PLOT
from utils.read_csv import read_from_dtw_file

COLORS = ['maroon', 'blue', 'green', 'orange']
COLORS = XKCD_COLORS_LIST


def pid_analysis(big_matrix_df: DataFrame, aoi_analysis_df: DataFrame, max_clusters: int = 20, pid_matrix_plot_file: str = PID_DISTANCE_PLOT):
    matrix_df = create_pid_matrix(big_matrix_df, fill_self_distances=0)
    cluster_df = get_best_fit_clusters(matrix_df, get_heirarchical_clusters_by_pid, max_clusters=max_clusters)
    plot_pid_matrix_with_clusters(matrix_df, cluster_df, colors=COLORS, to_file=pid_matrix_plot_file)
    responses_df = get_pid_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, PID, COLORS, PID_PERCENT_LIES_PLOT)
    plot_dwell_times_for_clusters(responses_df, PID, COLORS, PID_DWELL_TIMES_PLOT)
    plot_n_transitions_for_clusters(responses_df, PID, COLORS, PID_N_TRANSITIONS_PLOT)


def trial_id_analysis(big_matrix_df: DataFrame, aoi_analysis_df: DataFrame, max_clusters: int = 20, trial_id_matrix_plot_file: str = TRIAL_DISTANCE_PLOT):
    matrix_df = create_trial_id_matrix(big_matrix_df, fill_self_distances=0)
    cluster_df = get_best_fit_clusters(matrix_df, get_heirarchical_clusters_by_trial_id, max_clusters=max_clusters)
    plot_trial_id_matrix_with_clusters(matrix_df, cluster_df, colors=COLORS, to_file=trial_id_matrix_plot_file)
    responses_df = get_trial_id_response_stats_for_clusters(cluster_df, aoi_analysis_df)
    plot_percent_lies_for_clusters(responses_df, TRIAL_ID, COLORS, TRIAL_PERCENT_LIES_PLOT)
    plot_dwell_times_for_clusters(responses_df, TRIAL_ID, COLORS, TRIAL_DWELL_TIMES_PLOT)
    plot_n_transitions_for_clusters(responses_df, TRIAL_ID, COLORS, TRIAL_N_TRANSITIONS_PLOT)


def analyse(distance_df: DataFrame, aoi_analysis_df: DataFrame):
    big_matrix_df = create_big_matrix(distance_df)
    pid_analysis(big_matrix_df, aoi_analysis_df, max_clusters=20)
    trial_id_analysis(big_matrix_df, aoi_analysis_df, max_clusters=20)


if __name__ == "__main__":
    distance_df = read_from_dtw_file(DTW_PEN_Z_CSV)
    aoi_analysis_df = read_from_analysis_file()
    analyse(distance_df, aoi_analysis_df)
