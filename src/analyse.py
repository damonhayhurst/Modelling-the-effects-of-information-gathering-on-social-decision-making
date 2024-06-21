
from pandas.errors import SettingWithCopyWarning
from analyse.main import *
from analyse.response_analysis import *
from utils.paths import AOI_ANALYSIS_CSV, DTW_T_V2_CSV, KMEANS_DWELL_TIMES_PLOT, KMEANS_N_TRANSITIONS_PLOT, KMEANS_PERCENT_LIES_PLOT, PID_DISTANCE_PLOT, PID_DWELL_TIMES_PLOT, PID_N_TRANSITIONS_PLOT, PID_PERCENT_LIES_PLOT, TRIAL_DISTANCE_PLOT, TRIAL_DWELL_TIMES_PLOT, TRIAL_N_TRANSITIONS_PLOT, TRIAL_PERCENT_LIES_PLOT
import warnings

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)


def analyse():

    pid_dtw_analysis(
        input_distance_file=DTW_T_V2_CSV,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        pid_matrix_plot=PID_DISTANCE_PLOT,
        pid_percent_lies_plot=PID_PERCENT_LIES_PLOT,
        pid_dwell_times_plot=PID_DWELL_TIMES_PLOT,
        pid_n_transitions_plot=PID_N_TRANSITIONS_PLOT,
        percent_lies_by_pid_plot=PERCENT_LIES_BY_PID_PLOT,
    )

    trial_id_dtw_analysis(
        input_distance_file=DTW_T_V2_CSV,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        trial_id_matrix_plot=TRIAL_DISTANCE_PLOT,
        trial_id_percent_lies_plot=TRIAL_PERCENT_LIES_PLOT,
        trial_id_dwell_times_plot=TRIAL_DWELL_TIMES_PLOT,
        trial_id_n_transitions_plot=TRIAL_N_TRANSITIONS_PLOT,
        percent_lies_by_trial_id_plot=PERCENT_LIES_BY_TRIAL_ID_PLOT
    )

    # trial_count_dtw_analysis(
    #     input_distance_file=DTW_T_V2_CSV,
    #     input_aoi_analysis_file=AOI_ANALYSIS_CSV,
    #     trial_count_matrix_plot=TRIAL_COUNT_DISTANCE_PLOT,
    #     trial_count_percent_lies_plot=TRIAL_COUNT_PERCENT_LIES_PLOT,
    #     trial_count_dwell_times_plot=TRIAL_COUNT_DWELL_TIMES_PLOT,
    #     trial_count_n_transitions_plot=TRIAL_COUNT_N_TRANSITIONS_PLOT,
    # )

    # all_trial_dtw_analysis(
    #     input_distance_file=DTW_T_V2_CSV,
    #     input_aoi_analysis_file=AOI_ANALYSIS_CSV,
    #     all_trial_percent_lies_plot=ALL_TRIAL_PERCENT_LIES_PLOT,
    #     all_trial_dwell_times_plot=ALL_TRIAL_DWELL_TIMES_PLOT,
    #     all_trial_n_transitions_plot=ALL_TRIAL_N_TRANSITIONS_PLOT,
    # )

    kmeans_analysis(
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=KMEANS_PERCENT_LIES_PLOT,
        dwell_times_plot=KMEANS_DWELL_TIMES_PLOT,
        n_transitions_plot=KMEANS_N_TRANSITIONS_PLOT,
    )


if __name__ == "__main__":
    analyse()
