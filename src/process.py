import pandas as pd
from pandas import DataFrame
from dtw.main import do_dtw_processing
from preprocess.filtering import remove_5_percent_others, remove_95_percent_lies, remove_95_percent_truths, remove_dnf, remove_no_mouse_coords, remove_over_percent_no_mouse_coords, remove_practices
from preprocess.main import do_preprocessing
from utils.paths import AOI_ANALYSIS_CSV, AOIS_CSV, AVERAGE_ANALYSIS_CSV, DTW_NDIM_CSV, DTW_PEN_V1_CSV, DTW_PEN_Z_CSV, DTW_T_CSV, DTW_Z_V2_CSV, DWELL_TIMELINE_CSV, TRIAL_DISTANCE_PLOT, TRIAL_INDEX_CSV, YOUNG_ADULTS_1, YOUNG_ADULTS_2
import warnings

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

def do_filtering(input_df: DataFrame):
    return input_df.pipe(
        remove_practices, bypass=False
    ).pipe(
        remove_dnf
    ).pipe(
        remove_95_percent_lies
    ).pipe(
        remove_95_percent_truths
    ).pipe(
        remove_5_percent_others
    ).pipe(
        remove_no_mouse_coords, bypass=True
    ).pipe(
        remove_over_percent_no_mouse_coords, percent=85
    )

def process(input_files: str = [YOUNG_ADULTS_1, YOUNG_ADULTS_2]):
    do_preprocessing(
        input_data_files=input_files,
        output_trial_index_file=TRIAL_INDEX_CSV,
        output_aois_file=AOIS_CSV,
        output_aoi_analysis_file=AOI_ANALYSIS_CSV,
        output_avg_aoi_analysis_file=AVERAGE_ANALYSIS_CSV,
        output_dwell_timeline_file=DWELL_TIMELINE_CSV,
        custom_filtering=do_filtering,
        # bypass=True
    )

    do_dtw_processing(
        # from_file=DTW_Z_V2_CSV,
        input_dwell_file=DWELL_TIMELINE_CSV,
        input_aoi_analysis_file=AOI_ANALYSIS_CSV,
        output_dtw_file=DTW_T_CSV,
    )


if __name__ == '__main__':
    process()
