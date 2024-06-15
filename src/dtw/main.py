from dtw.dtw import get_dtw_distance
from utils.paths import AOI_ANALYSIS_CSV, DTW_Z_V2_CSV, DWELL_TIMELINE_CSV, TRIAL_DISTANCE_PLOT
from utils.read_csv import read_from_analysis_file, read_from_dtw_file, read_from_dwell_file


def do_dtw_processing(from_file: str = None,
                      input_dwell_file: str = DWELL_TIMELINE_CSV,
                      input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
                      output_dtw_file: str = DTW_Z_V2_CSV):

    if from_file is not None:
        distance_df = read_from_dtw_file(from_file)
    else:
        analysis_df = read_from_analysis_file(input_aoi_analysis_file)
        dwell_df = read_from_dwell_file(input_dwell_file)
        distance_df = get_dtw_distance(dwell_df, analysis_df, to_file=output_dtw_file)
    return distance_df
