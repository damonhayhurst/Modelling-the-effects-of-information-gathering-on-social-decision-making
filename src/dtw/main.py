import warnings
from joblib import Parallel, delayed
import numpy
import pandas as pd
from dtw.dtw import get_dtw_distance, get_t_series_df, get_t_series_dwell_sequence, get_t_series_dwell_sequences, get_t_series_sequences_and_trials
from utils.columns import CLUSTER, DISTANCE, PID_1, PID_2, SELECTED_AOI_1, SELECTED_AOI_2, TRIAL_COUNT, TRIAL_COUNT_1, TRIAL_COUNT_2, TRIAL_ID_1, TRIAL_ID_2
from utils.display import display
from utils.paths import AOI_ANALYSIS_CSV, DTW_T_CSV, DTW_T_V2_CSV, DTW_Z_V2_CSV, DWELL_TIMELINE_CSV, TIME_SERIES_KMEANS_10_CLUSTER_CSV, TIME_SERIES_KMEANS_2_CLUSTER_CSV, TIME_SERIES_KMEANS_3_CLUSTER_CSV, TIME_SERIES_KMEANS_4_CLUSTER_CSV, TIME_SERIES_KMEANS_5_CLUSTER_CSV, TIME_SERIES_KMEANS_6_CLUSTER_CSV, TIME_SERIES_KMEANS_7_CLUSTER_CSV, TIME_SERIES_KMEANS_8_CLUSTER_CSV, TIME_SERIES_KMEANS_9_CLUSTER_CSV, TRIAL_DISTANCE_PLOT
from utils.read_csv import read_from_analysis_file, read_from_dtw_file, read_from_dwell_file
import sktime
import numpy as np
from dtaidistance.clustering import KMeans

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


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


def do_time_series_kmeans_processing(
        input_dwell_file: str = DWELL_TIMELINE_CSV,
        input_aoi_analysis_file: str = AOI_ANALYSIS_CSV,
        bypass: bool = False):

    if bypass:
        return

    analysis_df = read_from_analysis_file(input_aoi_analysis_file)
    dwell_df = read_from_dwell_file(input_dwell_file)

    t_series_sequences, _ = get_t_series_sequences_and_trials(dwell_df, analysis_df)
    
    def k_cluster_processing(k: int, to_file: str = None):

        def monitor_distances(cluster_distances, stopped):
            df = pd.DataFrame(cluster_distances, columns=['Cluster', 'Distance'])
            centroids = df.groupby('Cluster')['Distance'].mean()
            inertia = 0
            for cluster, distance in cluster_distances:
                inertia += (distance - centroids[cluster]) ** 2

            print(k, inertia)

            return True

        cluster_indices, num_iter = KMeans(k).fit_fast(t_series_sequences, monitor_distances=monitor_distances)

        df = pd.DataFrame({
            CLUSTER: 0
        }, index=analysis_df.index).astype({CLUSTER: 'int64'})

        for cluster in cluster_indices.keys():
            for i in list(cluster_indices[cluster]):
                df.loc[df.index[i], CLUSTER] = cluster + 1

        df.to_csv(to_file)

    k_cluster_processing(8, TIME_SERIES_KMEANS_8_CLUSTER_CSV)



def _add_trial_id(distance_df: pd.DataFrame, aoi_analysis_df: pd.DataFrame, output_dtw_file: str = DTW_T_V2_CSV):

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
