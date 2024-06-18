import pandas as pd
from dtw.dtw import get_dtw_distance
from utils.columns import DISTANCE, PID_1, PID_2, SELECTED_AOI_1, SELECTED_AOI_2, TRIAL_COUNT, TRIAL_COUNT_1, TRIAL_COUNT_2, TRIAL_ID_1, TRIAL_ID_2
from utils.display import display
from utils.paths import AOI_ANALYSIS_CSV, DTW_T_CSV, DTW_T_V2_CSV, DTW_Z_V2_CSV, DWELL_TIMELINE_CSV, TRIAL_DISTANCE_PLOT
from utils.read_csv import read_from_analysis_file, read_from_dtw_file, read_from_dwell_file


def do_dtw_processing(from_file: str = None,
                      input_dwell_file: str = DWELL_TIMELINE_CSV,
                      input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                      output_dtw_file: str = DTW_Z_V2_CSV,
                      bypass: bool = False):

    if bypass:
        return

    if from_file is not None:
        distance_df = read_from_dtw_file(from_file)
    else:
        analysis_df = read_from_analysis_file(input_aoi_analysis_file)
        dwell_df = read_from_dwell_file(input_dwell_file)
        distance_df = get_dtw_distance(dwell_df, analysis_df, to_file=output_dtw_file)
    return distance_df


def _add_trial_id(distance_df: pd.DataFrame, aoi_analysis_df : pd.DataFrame, output_dtw_file: str = DTW_T_V2_CSV):

    # Extract index levels for PID and TRIAL_ID
    pid_1_values = distance_df.index.get_level_values(PID_1)
    trial_id_1_values = distance_df.index.get_level_values(TRIAL_ID_1)
    pid_2_values = distance_df.index.get_level_values(PID_2)
    trial_id_2_values = distance_df.index.get_level_values(TRIAL_ID_2)

    # Create a MultiIndex for selection
    index_1 = pd.MultiIndex.from_arrays([pid_1_values, trial_id_1_values], names=[PID_1, TRIAL_ID_1])
    index_2 = pd.MultiIndex.from_arrays([pid_2_values, trial_id_2_values], names=[PID_2, TRIAL_ID_2])

    # Select the required data from analysis_df and assign it to distance_df
    distance_df[TRIAL_COUNT_1] = aoi_analysis_df.loc[index_1, TRIAL_COUNT].values
    distance_df[TRIAL_COUNT_2] = aoi_analysis_df.loc[index_2, TRIAL_COUNT].values

    if output_dtw_file:
        distance_df.reset_index(inplace=True)
        display(distance_df)
        distance_df = distance_df[[PID_1, TRIAL_ID_1, TRIAL_COUNT_1, PID_2, TRIAL_ID_2, TRIAL_COUNT_2, SELECTED_AOI_1, SELECTED_AOI_2, DISTANCE]]
        distance_df.to_csv(output_dtw_file, index=False)
