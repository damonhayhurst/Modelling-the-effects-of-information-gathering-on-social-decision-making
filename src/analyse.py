
from pandas.errors import SettingWithCopyWarning
from analyse.main import *
from analyse.response_analysis import *
from dtw.main import do_time_series_kmeans_processing
from utils.paths import *
import warnings

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

distance_file = DTW_T_V2_CSV

def save_to_file(do_save): return lambda path: path if do_save else None

def do_dtw_analysis(save: bool = True):
    
    save = save_to_file(save)

    # pid_dtw_analysis(
    #     input_distance_file=distance_file,
    #     input_aoi_analysis_file=AOI_ANALYSIS_CSV,
    #     pid_matrix_plot=save(PID_DISTANCE_PLOT),
    #     pid_percent_lies_plot=save(PID_PERCENT_LIES_PLOT),
    #     pid_dwell_times_plot=save(PID_DWELL_TIMES_PLOT),
    #     pid_n_transitions_plot=save(PID_N_TRANSITIONS_PLOT),
    #     percent_lies_by_pid_plot=save(PERCENT_LIES_BY_PID_PLOT),
    # )

    # trial_id_dtw_analysis(
    #     input_distance_file=distance_file,
    #     input_aoi_analysis_file=AOI_ANALYSIS_CSV,
    #     trial_id_matrix_plot=save(TRIAL_DISTANCE_PLOT),
    #     trial_id_percent_lies_plot=save(TRIAL_PERCENT_LIES_PLOT),
    #     trial_id_dwell_times_plot=save(TRIAL_DWELL_TIMES_PLOT),
    #     trial_id_n_transitions_plot=save(TRIAL_N_TRANSITIONS_PLOT),
    #     percent_lies_by_trial_id_plot=save(PERCENT_LIES_BY_TRIAL_ID_PLOT)
    # )

    all_trial_dtw_analysis(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        all_trial_percent_lies_plot=save(ALL_TRIAL_PERCENT_LIES_PLOT),
        all_trial_dwell_times_plot=save(ALL_TRIAL_DWELL_TIMES_PLOT),
        all_trial_n_transitions_plot=save(ALL_TRIAL_N_TRANSITIONS_PLOT),
        all_trial_n_trials_plot=save(ALL_TRIAL_N_TRIALS_PLOT),
        # max_clusters=2
        # n_clusters=10
    )


def do_kmeans_analysis(save: bool = True):

    save = save_to_file(save)

    kmeans_analysis(
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=save(KMEANS_PERCENT_LIES_PLOT),
        n_trials_plot=save(KMEANS_N_TRIALS_PLOT),
        dwell_times_plot=save(KMEANS_DWELL_TIMES_PLOT),
        n_transitions_plot=save(KMEANS_N_TRANSITIONS_PLOT),
        correlations_plot=save(KMEANS_CORRELATIONS_PLOT),
        percent_lies_by_pid_plot=save(KMEANS_PERCENT_LIES_BY_PID_PLOT),
        n_trials_by_pid_plot=save(KMEANS_N_TRIALS_BY_PID_PLOT),
        columns=[SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH, N_TRANSITIONS, SELF_GAIN, OTHER_LOSS, UNIQUE_AOIS, RT]
    )

def do_time_series_kmeans_analysis(save: bool = True):
     
    save = save_to_file(save)

    input_cluster_files = [TIME_SERIES_KMEANS_2_CLUSTER_CSV, 
                           TIME_SERIES_KMEANS_3_CLUSTER_CSV, 
                           TIME_SERIES_KMEANS_4_CLUSTER_CSV, 
                           TIME_SERIES_KMEANS_5_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_6_CLUSTER_CSV]

    time_series_kmeans_analysis(
        input_cluster_files=input_cluster_files,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        input_distance_file=distance_file,
        percent_lies_plot=save(TS_KMEANS_PERCENT_LIES_PLOT),
        dwell_times_plot=save(TS_KMEANS_DWELL_TIMES_PLOT),
        n_transitions_plot=save(TS_KMEANS_N_TRANSITIONS_PLOT),
        n_trials_plot=save(TS_KMEANS_N_TRIALS_PLOT),
        percent_lies_by_pid_plot=save(TS_KMEANS_PERCENT_LIES_BY_PID_PLOT),
        n_trials_by_pid_plot=save(TS_KMEANS_N_TRIALS_BY_PID_PLOT),
    )

def do_kmedoids_analysis(save: bool = True):

    save = save_to_file(save)

    kmedoids_dtw_analysis(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=save(KMEDOIDS_PERCENT_LIES_PLOT),
        dwell_times_plot=save(KMEDOIDS_DWELL_TIMES_PLOT),
        n_transitions_plot=save(KMEDOIDS_N_TRANSITIONS_PLOT),
        percent_lies_by_pid_plot=save(KMEDOIDS_PERCENT_LIES_BY_PID_PLOT),
        n_trials_by_pid_plot=save(KMEDOIDS_N_TRIALS_BY_PID_PLOT)
    )


def do_dbscan_analysis(save: bool = True):

    save = save_to_file(save)

    dbscan_dtw_analysis(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=save(DBSCAN_PERCENT_LIES_PLOT),
        dwell_times_plot=save(DBSCAN_DWELL_TIMES_PLOT),
        n_transitions_plot=save(DBSCAN_N_TRANSITIONS_PLOT),
        n_trials_plot=save(DBSCAN_N_TRIALS_BY_PID_PLOT),
        percent_lies_by_pid_plot=save(DBSCAN_PERCENT_LIES_BY_PID_PLOT),
        n_trials_by_pid_plot=save(DBSCAN_N_TRIALS_BY_PID_PLOT)
    )


def do_response_analysis(save: bool = True):

    save = save_to_file(save)

    response_analysis(
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        input_trial_index_file=TRIAL_INDEX_CSV,
        self_lie_distribution_plot=save(SELF_LIE_DISTRIBUTION_PLOT),
        self_true_distribution_plot=save(SELF_TRUE_DISTRIBUTION_PLOT),
        other_lie_distribution_plot=save(OTHER_LIE_DISTRIBUTION_PLOT),
        other_true_distribution_plot=save(OTHER_TRUTH_DISTRIBUTION_PLOT),
        n_trials_by_pid_plot=save(N_TRIALS_BY_PID_PLOT),
        percent_lies_by_pid_plot=save(OVERALL_PID_PERCENT_LIES_PLOT),
        percent_lies_by_trial_id_plot=save(OVERALL_TRIAL_ID_PERCENT_LIES_PLOT),
        net_gain_lie_plot=save(NET_GAIN_LIE_PLOT)
    )

def do_proximal_analysis(save: bool = True):

    proximal_analysis(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV
    )

if __name__ == "__main__":
    # with plt.ioff():
    # with plt.ion():
        do_dtw_analysis(save=True)
        # do_kmeans_analysis(save=False)
        # do_kmedoids_analysis(save=False)
        # do_response_analysis(save=True)
        # do_dbscan_analysis(save=False)
        # do_proximal_analysis(save=False)
        # do_time_series_kmeans_analysis(save=False)
