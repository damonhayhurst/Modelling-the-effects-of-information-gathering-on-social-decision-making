
from pandas import DataFrame
from pandas.errors import SettingWithCopyWarning
from analyse.main import do_dtw_analysis, do_kmeans_analysis
from analyse.response_analysis import *
from utils.read_csv import read_from_analysis_file
from utils.paths import AOI_ANALYSIS_CSV, DTW_PEN_Z_CSV, DTW_T_CSV, DTW_T_V2_CSV, KMEANS_DWELL_TIMES_PLOT, KMEANS_N_TRANSITIONS_PLOT, KMEANS_PERCENT_LIES_PLOT, PID_DISTANCE_PLOT, PID_DWELL_TIMES_PLOT, PID_N_TRANSITIONS_PLOT, PID_PERCENT_LIES_PLOT, TRIAL_COUNT_DISTANCE_PLOT, TRIAL_COUNT_DWELL_TIMES_PLOT, TRIAL_COUNT_N_TRANSITIONS_PLOT, TRIAL_COUNT_PERCENT_LIES_PLOT, TRIAL_DISTANCE_PLOT, TRIAL_DWELL_TIMES_PLOT, TRIAL_N_TRANSITIONS_PLOT, TRIAL_PERCENT_LIES_PLOT
from utils.read_csv import read_from_dtw_file
import warnings

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)

def analyse(do_dtw: bool = True, do_kmeans: bool = True):
    do_dtw_analysis(
        input_distance_file=DTW_T_V2_CSV,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        pid_matrix_plot=PID_DISTANCE_PLOT,
        pid_percent_lies_plot=PID_PERCENT_LIES_PLOT,
        pid_dwell_times_plot=PID_DWELL_TIMES_PLOT,
        pid_n_transitions_plot=PID_N_TRANSITIONS_PLOT,
        trial_id_matrix_plot=TRIAL_DISTANCE_PLOT,
        trial_id_percent_lies_plot=TRIAL_PERCENT_LIES_PLOT,
        trial_id_dwell_times_plot=TRIAL_DWELL_TIMES_PLOT,
        trial_id_n_transitions_plot=TRIAL_N_TRANSITIONS_PLOT,
        trial_count_matrix_plot=TRIAL_COUNT_DISTANCE_PLOT,
        trial_count_percent_lies_plot=TRIAL_COUNT_PERCENT_LIES_PLOT,
        trial_count_dwell_times_plot=TRIAL_COUNT_DWELL_TIMES_PLOT,
        trial_count_n_transitions_plot=TRIAL_COUNT_N_TRANSITIONS_PLOT,
        bypass=not do_dtw
    )

    do_kmeans_analysis(
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        percent_lies_plot=KMEANS_PERCENT_LIES_PLOT,
        dwell_times_plot=KMEANS_DWELL_TIMES_PLOT,
        n_transitions_plot=KMEANS_N_TRANSITIONS_PLOT,
        bypass=not do_kmeans
    )


if __name__ == "__main__":
    analyse(do_dtw=True, do_kmeans=False)
