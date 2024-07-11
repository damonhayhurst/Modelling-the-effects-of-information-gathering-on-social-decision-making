import pandas as pd
from pandas import DataFrame
from dtw.main import do_dtw_processing, do_time_series_kmeans_processing
from preprocess.filtering import remove_percent_others, remove_percent_lies, remove_percent_truths, remove_dnf, remove_no_mouse_coords, remove_over_percent_no_mouse_coords, remove_practices
from preprocess.main import do_preprocessing
from utils.paths import AOI_ANALYSIS_CSV, AOIS_CSV, AVERAGE_ANALYSIS_CSV, DTW_NDIM_CSV, DTW_PEN_V1_CSV, DTW_PEN_Z_CSV, DTW_T_CSV, DTW_T_WITH_DIFF_CSV, DTW_Z_V2_CSV, DWELL_TIMELINE_CSV, TRIAL_DISTANCE_PLOT, TRIAL_INDEX_CSV, TRIAL_INDEX_GAINS_PLOT, YOUNG_ADULTS_1, YOUNG_ADULTS_2
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


def filtering(input_df: DataFrame):
    return input_df.pipe(
        remove_practices, bypass=False
    ).pipe(
        remove_dnf
    ).pipe(
        remove_percent_lies, percent=95
    ).pipe(
        remove_percent_truths, percent=95
    ).pipe(
        remove_percent_others, percent=5
    ).pipe(
        remove_no_mouse_coords, bypass=True
    ).pipe(
        remove_over_percent_no_mouse_coords, percent=85
    )


def process(input_files: str = [YOUNG_ADULTS_1, YOUNG_ADULTS_2]):
    do_preprocessing(
        input_data_files=input_files,
        output_trial_index_gains_plot=TRIAL_INDEX_GAINS_PLOT,
        output_trial_index_file=TRIAL_INDEX_CSV,
        output_aois_file=AOIS_CSV,
        output_aoi_analysis_file=AOI_ANALYSIS_CSV,
        output_avg_aoi_analysis_file=AVERAGE_ANALYSIS_CSV,
        output_dwell_timeline_file=DWELL_TIMELINE_CSV,
        custom_filtering=filtering,
    )

def dtw_process():
    do_dtw_processing(
        input_dwell_file=DWELL_TIMELINE_CSV,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        output_dtw_file=DTW_T_WITH_DIFF_CSV,
        bypass=True
    )

    do_time_series_kmeans_processing(
        input_dwell_file=DWELL_TIMELINE_CSV,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
    )


if __name__ == '__main__':
    # process()
    dtw_process()
