
from pandas.errors import SettingWithCopyWarning
from analyse.main import do_analyses
from analyse.response_analysis import *
from dtw.main import do_time_series_kmeans_processing
from utils.paths import *
import warnings
import argparse

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

distance_file = DTW_T_WITH_DIFF_WITH_SMOOTH_CSV
analysis_file = AOI_ANALYSIS_V2_CSV

def get_pid_dtw_analysis_params(save: bool = False):

    return dict(
        input_distance_file=distance_file,
        input_aoi_analysis_file=analysis_file,
        pid_matrix_plot=PID_DISTANCE_PLOT if save else None,
        pid_percent_lies_plot=PID_PERCENT_LIES_PLOT if save else None,
        pid_dwell_times_plot=PID_DWELL_TIMES_PLOT if save else None,
        pid_n_transitions_plot=PID_N_TRANSITIONS_PLOT if save else None,
        percent_lies_by_pid_plot=PERCENT_LIES_BY_PID_PLOT if save else None,
        percent_lies_gain_cluster_plot=PERCENT_LIES_GAIN_CLUSTER_PLOT if save else None,
        n_transitions_gain_cluster_plot=N_TRANSITION_GAIN_CLUSTER_PLOT if save else None,
        self_lie_gain_cluster_plot=SELF_LIE_GAIN_CLUSTER_PLOT if save else None,
        self_true_gain_cluster_plot=SELF_TRUE_GAIN_CLUSTER_PLOT if save else None,
        other_lie_gain_cluster_plot=OTHER_LIE_GAIN_CLUSTER_PLOT if save else None,
        other_truth_gain_cluster_plot=OTHER_TRUTH_GAIN_CLUSTER_PLOT if save else None,
    )

def get_response_analysis_params(save: bool = False):
    return dict(
        input_aoi_analysis_file=AOI_ANALYSIS_V2_CSV,
        input_trial_index_file=TRIAL_INDEX_CSV,
        n_trials_by_pid_plot=N_TRIALS_BY_PID_PLOT if save else None,
        percent_lies_by_pid_plot=OVERALL_PID_PERCENT_LIES_PLOT if save else None,
        percent_lies_by_trial_id_plot=OVERALL_TRIAL_ID_PERCENT_LIES_PLOT if save else None,
        net_gain_lie_plot=NET_GAIN_LIE_PLOT if save else None,
        net_loss_lie_plot=NET_LOSS_LIE_PLOT if save else None,
        net_gain_loss_lie_plot=NET_GAIN_LOSS_LIE_PLOT if save else None,
        avg_dwell_per_gain_plot=AVG_DWELL_PER_GAIN_PLOT if save else None,
        avg_n_transition_per_gain_plot=N_TRANSITION_PER_GAIN_PLOT if save else None,
        avg_dwell_per_loss_plot=AVG_DWELL_PER_LOSS_PLOT if save else None,
        avg_n_transition_per_loss_plot=N_TRANSITION_PER_LOSS_PLOT if save else None,
        output_trial_index_gains_plot=TRIAL_INDEX_GAINS_PLOT if save else None
    )


def get_trial_id_dtw_analysis_params(save: bool = False):
    return dict(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        trial_id_matrix_plot=TRIAL_DISTANCE_PLOT if save else None,
        trial_id_percent_lies_plot=TRIAL_PERCENT_LIES_PLOT if save else None,
        trial_id_dwell_times_plot=TRIAL_DWELL_TIMES_PLOT if save else None,
        trial_id_n_transitions_plot=TRIAL_N_TRANSITIONS_PLOT if save else None,
        percent_lies_by_trial_id_plot=PERCENT_LIES_BY_TRIAL_ID_PLOT if save else None
    )

def get_all_trial_dtw_analysis_params(save: bool = False):
    return dict(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        all_trial_percent_lies_plot=ALL_TRIAL_PERCENT_LIES_PLOT if save else None,
        all_trial_dwell_times_plot=ALL_TRIAL_DWELL_TIMES_PLOT if save else None,
        all_trial_n_transitions_plot=ALL_TRIAL_N_TRANSITIONS_PLOT if save else None,
        all_trial_n_trials_plot=ALL_TRIAL_N_TRIALS_PLOT if save else None,
        all_trial_percent_lie_by_pid_plot=ALL_TRIAL_PERCENT_LIES_BY_PID_PLOT if save else None,
        all_trial_n_trials_by_pid_plot=ALL_TRIAL_N_TRIALS_BY_PID_PLOT if save else None,
        all_trial_gain_of_ten_by_pid_plot=ALL_TRIAL_GAIN_OF_TEN_BY_PID_PLOT if save else None,
        all_trial_gain_under_ten_by_pid_plot=ALL_TRIAL_GAIN_UNDER_TEN_BY_PID_PLOT if save else None
        # max_clusters=2
        # n_clusters=10
    )


def get_kmeans_analysis_params(save: bool = True):

    return dict(
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=KMEANS_PERCENT_LIES_PLOT if save else None,
        n_trials_plot=KMEANS_N_TRIALS_PLOT if save else None,
        dwell_times_plot=KMEANS_DWELL_TIMES_PLOT if save else None,
        n_transitions_plot=KMEANS_N_TRANSITIONS_PLOT if save else None,
        correlations_plot=KMEANS_CORRELATIONS_PLOT if save else None,
        percent_lies_by_pid_plot=KMEANS_PERCENT_LIES_BY_PID_PLOT if save else None,
        n_trials_by_pid_plot=KMEANS_N_TRIALS_BY_PID_PLOT if save else None,
        gain_of_ten_by_pid_plot=KMEANS_GAIN_OF_TEN_BY_PID_PLOT if save else None,
        gain_under_ten_by_pid_plot=KMEANS_GAIN_UNDER_TEN_BY_PID_PLOT if save else None,
        columns=[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH, N_TRANSITIONS]
    )

def get_time_series_kmeans_params(save: bool = False):
     

    input_cluster_files = [TIME_SERIES_KMEANS_2_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_3_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_4_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_5_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_6_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_7_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_8_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_9_CLUSTER_CSV,
                           TIME_SERIES_KMEANS_10_CLUSTER_CSV]

    return dict(
        input_cluster_files=input_cluster_files,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        input_distance_file=distance_file,
        percent_lies_plot=TS_KMEANS_PERCENT_LIES_PLOT if save else None,
        dwell_times_plot=TS_KMEANS_DWELL_TIMES_PLOT if save else None,
        n_transitions_plot=TS_KMEANS_N_TRANSITIONS_PLOT if save else None,
        n_trials_plot=TS_KMEANS_N_TRIALS_PLOT if save else None,
        percent_lies_by_pid_plot=TS_KMEANS_PERCENT_LIES_BY_PID_PLOT if save else None,
        n_trials_by_pid_plot=TS_KMEANS_N_TRIALS_BY_PID_PLOT if save else None,
        gain_of_ten_by_pid_plot=TS_KMEANS_GAIN_OF_TEN_BY_PID_PLOT if save else None,
        gain_under_ten_by_pid_plot=TS_KMEANS_GAIN_UNDER_TEN_BY_PID_PLOT if save else None
    )

def get_kmedoids_analysis_params(save: bool = True):


    return dict(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=KMEDOIDS_PERCENT_LIES_PLOT if save else None,
        dwell_times_plot=KMEDOIDS_DWELL_TIMES_PLOT if save else None,
        n_transitions_plot=KMEDOIDS_N_TRANSITIONS_PLOT if save else None,
        percent_lies_by_pid_plot=KMEDOIDS_PERCENT_LIES_BY_PID_PLOT if save else None,
        n_trials_by_pid_plot=KMEDOIDS_N_TRIALS_BY_PID_PLOT if save else None
    )


def get_dbscan_analysis_params(save: bool = True):


    return dict(
        input_distance_file=distance_file,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=DBSCAN_PERCENT_LIES_PLOT if save else None,
        dwell_times_plot=DBSCAN_DWELL_TIMES_PLOT if save else None,
        n_transitions_plot=DBSCAN_N_TRANSITIONS_PLOT if save else None,
        n_trials_plot=DBSCAN_N_TRIALS_BY_PID_PLOT if save else None,
        percent_lies_by_pid_plot=DBSCAN_PERCENT_LIES_BY_PID_PLOT if save else None,
        n_trials_by_pid_plot=DBSCAN_N_TRIALS_BY_PID_PLOT if save else None,
        eps=52.1
    )


    

def get_proximal_analysis_params(save: bool = False):
    return dict(
        input_distance_file=distance_file
    )


def get_distribution_analysis_params(save: bool = False):
    return dict(
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        input_dwell_timeline_file=DWELL_TIMELINE_CSV,
        dwell_distribution_plot=DWELL_DISTRIBUTION_PLOT if save else None,
        rt_distribution_plot=RT_DISTRIBUTION_PLOT if save else None,
        n_transition_distribution_plot=N_TRANSITIONS_DISTRIBUTION_PLOT if save else None
    ) 


def get_descriptives_params(save: bool = False):
    return dict(
        input_aoi_analysis_file=AOI_ANALYSIS_CSV
    )

def main(args):
    plt.ioff() if args.no_plt else plt.ion()

    do_analyses(
        do_save=args.save,
        pid_dtw=get_pid_dtw_analysis_params,
        response=get_response_analysis_params,
        distribution=get_distribution_analysis_params,
        proximal=get_proximal_analysis_params,
        trial_id_dtw=get_trial_id_dtw_analysis_params,
        all_trials_dtw=get_all_trial_dtw_analysis_params,
        kmeans=get_kmeans_analysis_params,
        ts_kmeans=get_time_series_kmeans_params,
        kmedoids=get_kmedoids_analysis_params,
        dbscan=get_dbscan_analysis_params,
        descriptives=get_descriptives_params
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plt", action="store_false",  help="Don't plot")
    parser.add_argument("--save", action="store_true",  help="Save plots to file")
    args = parser.parse_args()
    main(args)
