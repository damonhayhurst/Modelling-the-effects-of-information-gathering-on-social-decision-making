from typing import Callable
from pandas import DataFrame
from analyse.response_analysis import plot_gains_by_trial_id
from preprocess.aoi import determine_aoi
from preprocess.aoi_analysis import create_aoi_analysis, create_average_analysis
from preprocess.dwell import create_dwell_timeline
from preprocess.filtering import do_filtering, print_stats
from preprocess.trial_id import add_gains, add_trial_id, calculate_gains_losses, create_trial_id_index
from utils.display import display
from utils.paths import AOI_ANALYSIS_CSV, AOIS_CSV, AVERAGE_ANALYSIS_CSV, DWELL_TIMELINE_CSV, TRIAL_INDEX_CSV, YOUNG_ADULTS_1, YOUNG_ADULTS_2
from utils.read_csv import read_from_aois_file, read_from_input_files


def do_preprocessing(input_data_files: str = [YOUNG_ADULTS_1, YOUNG_ADULTS_2],
                     output_trial_index_gains_plot: str = None,
                     output_trial_index_file: str = TRIAL_INDEX_CSV,
                     output_aois_file: str = AOIS_CSV,
                     output_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                     output_avg_aoi_analysis_file: str = AVERAGE_ANALYSIS_CSV,
                     output_dwell_timeline_file: str = DWELL_TIMELINE_CSV,
                     custom_filtering: Callable[[DataFrame], DataFrame] = None,
                     bypass: bool = False):
    if bypass:
        return

    df = read_from_input_files(input_data_files)
    print_stats(df)
    df = do_filtering(df) if custom_filtering is None else custom_filtering(df)
    print_stats(df)
    trial_index = create_trial_id_index(df, to_file=output_trial_index_file)
    gains_df = calculate_gains_losses(trial_index)
    
    df = add_trial_id(df, trial_index)
    df = add_gains(df, gains_df)
    aoi_df = determine_aoi(df, to_file=output_aois_file)
    display(aoi_df)
    print_stats(aoi_df)
    analysis_df = create_aoi_analysis(aoi_df, to_file=output_aoi_analysis_file)
    create_average_analysis(analysis_df, to_file=output_avg_aoi_analysis_file)
    create_dwell_timeline(aoi_df, to_file=output_dwell_timeline_file)


def do_dwell_preprocessing(input_aois_file: str = None,
                           output_dwell_timeline_file: str = None):
    aoi_df = read_from_aois_file(input_aois_file)
    create_dwell_timeline(aoi_df, to_file=output_dwell_timeline_file)
