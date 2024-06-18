import math
from pandas import DataFrame, Series
from utils.masks import get_is_aoi
from utils.paths import DTW_Z_V2_CSV
from utils.display import display
from utils.read_csv import read_from_analysis_file, read_from_dwell_file
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.columns import *
from dtaidistance import dtw, dtw_ndim
from dtaidistance.innerdistance import CustomInnerDist, SquaredEuclidean
from itertools import combinations
from scipy.stats import zscore
from typing import List, Tuple

DWELL_DICT = {column: index for index, column in enumerate([np.nan] + DWELL_COLUMNS)}

def get_combinations_of_trials(df: DataFrame) -> List[Tuple[int, int]]:
    trials = df.index.unique()
    return list(combinations(trials, 2))


def get_dtw_distance(dwell_df: DataFrame, analysis_df: DataFrame, z_norm: bool = True, to_file: str = None) -> DataFrame:
    t_series = get_t_series_dwell_sequences(dwell_df, analysis_df)
    t_series_trials = [trial for trial in t_series.keys()]
    t_series_sequences = [t_series[trial] for trial in t_series_trials]
    dtws = dtw_ndim.distance_matrix_fast(t_series_sequences, only_triu=True)
    distances = {}
    idx = 0
    for x in range(0, len(dtws)):
        for y in range(0, len(dtws[x])):
            if y is not np.nan:
                pid1, trial_id1 = t_series_trials[x]
                pid2, trial_id2 = t_series_trials[y]
                selected_aoi_1, selected_aoi_2 = analysis_df.loc[pid1, trial_id1][SELECTED_AOI], analysis_df.loc[pid2, trial_id2][SELECTED_AOI]
                trial_count_1, trial_count_2 = analysis_df.loc[pid1, trial_id1][TRIAL_COUNT], analysis_df.loc[pid2, trial_id2][TRIAL_COUNT]
                distances[idx] = {
                    PID_1: pid1, TRIAL_ID_1: trial_id1, TRIAL_COUNT_1: trial_count_1,
                    PID_2: pid2, TRIAL_ID_2: trial_id2, TRIAL_COUNT_2: trial_count_2,
                    SELECTED_AOI_1: selected_aoi_1, SELECTED_AOI_2: selected_aoi_2,
                    DISTANCE: dtws[x][y]
                }
                idx = idx + 1
    distance_df = DataFrame.from_dict(distances, orient='index')
    distance_df = distance_df.set_index([PID_1, TRIAL_ID_1, PID_2, TRIAL_ID_2])
    if to_file:
        save(distance_df, to_file)
    return distance_df


def get_dtw_distance_old(dwell_df: DataFrame, analysis_df: DataFrame, z_norm: bool = True, to_file: str = None) -> DataFrame:
    trial_combs = get_combinations_of_trials(analysis_df)
    distances = {}
    t_series = get_t_series_dwell_sequences(dwell_df, analysis_df)
    for idx, ((pid1, trial_id1), (pid2, trial_id2)) in enumerate(trial_combs):
        # distance = get_ndim_distance(dwell_df.loc[(pid1, trial_id1)], dwell_df.loc[(pid2, trial_id2)])
        # distance = get_multivariate_distance(dwell_df.loc[(pid1, trial_id1)], dwell_df.loc[(pid2, trial_id2)], AOIPenaltySquareEuclidean(), z_norm)
        # distance = get_distance(dwell_df.loc[(pid1, trial_id1)][DWELL_TIME], dwell_df.loc[(pid2, trial_id2)][DWELL_TIME])
        trial1, trial2 = t_series[(pid1, trial_id2)], t_series[(pid2, trial_id2)]
        distance = dtw_ndim.distance_fast(trial1, trial2)
        selected_aoi_1, selected_aoi_2 = analysis_df.loc[pid1, trial_id1][SELECTED_AOI], analysis_df.loc[pid2, trial_id2][SELECTED_AOI]
        trial_count_1, trial_count_2 = analysis_df.loc[pid1, trial_id1][SELECTED_AOI], analysis_df.loc[pid2, trial_id2][SELECTED_AOI]
        distances[idx] = {
            PID_1: pid1, TRIAL_ID_1: trial_id1,
            PID_2: pid2, TRIAL_ID_2: trial_id2,
            SELECTED_AOI_1: selected_aoi_1, SELECTED_AOI_2: selected_aoi_2,
            DISTANCE: distance
        }
        print("%s of %s" % (idx, len(trial_combs)))
    distance_df = DataFrame.from_dict(distances, orient='index')
    distance_df = distance_df.set_index([PID_1, TRIAL_ID_1, PID_2, TRIAL_ID_2])
    if to_file:
        save(distance_df, to_file)
    return distance_df

def get_ndim_distance(trial1: DataFrame, trial2: DataFrame, z_norm: bool = False):
    idx1, idx2 = trial1.index, trial2.index
    trial1.loc[idx1, SELF_LIE], trial2.loc[idx2, SELF_LIE] = get_is_aoi(trial1, SELF_LIE).astype(int), get_is_aoi(trial2, SELF_LIE).astype(int)
    trial1.loc[idx1, SELF_TRUE], trial2.loc[idx2, SELF_TRUE] = get_is_aoi(trial1, SELF_TRUE).astype(int), get_is_aoi(trial2, SELF_TRUE).astype(int)
    trial1.loc[idx1, OTHER_LIE], trial2.loc[idx2, OTHER_LIE] = get_is_aoi(trial1, OTHER_LIE).astype(int), get_is_aoi(trial2, OTHER_LIE).astype(int)
    trial1.loc[idx1, OTHER_TRUTH], trial2.loc[idx2, OTHER_TRUTH] = get_is_aoi(trial1, OTHER_TRUTH).astype(int), get_is_aoi(trial2, OTHER_TRUTH).astype(int)
    if z_norm:
        trial1[DWELL_TIME] = zscore(trial1[DWELL_TIME])
        trial2[DWELL_TIME] = zscore(trial2[DWELL_TIME])
    trial1_tuples = trial1.loc[idx1][[DWELL_TIME, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]].itertuples(index=False, name=None)
    trial2_tuples = trial2.loc[idx2][[DWELL_TIME, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]].itertuples(index=False, name=None)
    trial1_ndim = np.array([[dwell, *aois] for dwell, *aois in trial1_tuples], dtype=object)
    trial2_ndim = np.array([[dwell, *aois] for dwell, *aois in trial2_tuples], dtype=object)
    return dtw_ndim.distance(trial1_ndim, trial2_ndim)

def get_t_series_dwell_sequences(dwell_df: DataFrame, analysis_df: DataFrame):
    trials = analysis_df.index.unique()
    return {trial: get_t_series_dwell_sequence(dwell_df.loc[trial]) for trial in trials}

def get_t_series_dwell_sequence(trial: DataFrame, z_norm: bool = False):
    idx = trial.index
    trial[DWELL_TIME] = pd.to_timedelta(trial[DWELL_TIME], "ms")
    trial[DWELL_TIME_MS] = (trial[DWELL_TIME].dt.microseconds).astype(int)
    trial.loc[idx, SELF_LIE] = get_is_aoi(trial, SELF_LIE).astype(int)
    trial.loc[idx, SELF_TRUE] = get_is_aoi(trial, SELF_TRUE).astype(int)
    trial.loc[idx, OTHER_LIE] = get_is_aoi(trial, OTHER_LIE).astype(int)
    trial.loc[idx, OTHER_TRUTH] = get_is_aoi(trial, OTHER_TRUTH).astype(int)
    t_series = []
    for time, *aois in trial.loc[idx][[DWELL_TIME_MS, SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]].itertuples(index=False, name=None):
        for t in range(time):
            t_series.append(aois)
    return np.array(t_series)


def get_multivariate_distance(dwell_series_1: DataFrame, dwell_series_2: DataFrame, dist_method: CustomInnerDist, z_norm: bool = False):
    dwell_series_1, dwell_series_2 = dwell_series_1[[DWELL_TIME, AOI]], dwell_series_2[[DWELL_TIME, AOI]]
    if z_norm:
        dwell_series_1[DWELL_TIME] = zscore(dwell_series_1[DWELL_TIME])
        dwell_series_2[DWELL_TIME] = zscore(dwell_series_2[DWELL_TIME])
    dwell_list_1 = np.array([(dur, DWELL_DICT[aoi]) for dur, aoi in dwell_series_1.itertuples(index=False, name=None)])
    dwell_list_2 = np.array([(dur, DWELL_DICT[aoi]) for dur, aoi in dwell_series_2.itertuples(index=False, name=None)])
    return dtw_ndim.distance(dwell_list_1, dwell_list_2, inner_dist=dist_method)


def get_distance(dwell_series_1: Series, dwell_series_2: Series, z_norm: bool = True):
    dwell_list_1, dwell_list_2 = dwell_series_1[DWELL_TIME].to_list(), dwell_series_2[DWELL_TIME].to_list()
    if z_norm:
        dwell_list_1, dwell_list_2 = zscore(dwell_list_1), zscore(dwell_list_2)
    print(dwell_list_1)
    print(type(dwell_list_1))
    return dtw.distance_fast(dwell_list_1, dwell_list_2)


class AOIPenaltySquareEuclidean(CustomInnerDist):


    @staticmethod
    def inner_dist(x, y):
        euc = x[0] - y[0]
        pen = x[0] - y[0] if x[1] != y[1] else 0
        return (euc + pen) ** 2

    @staticmethod
    def result(x):
        if np is not None and isinstance(x, np.ndarray):
            return np.sqrt(x)
        return math.sqrt(x)

    @staticmethod
    def inner_val(x):
        return x*x


def get_warping_path(dwell_series_1: Series, dwell_series_2: Series):
    dwell_list_1, dwell_list_2 = dwell_series_1.to_list(), dwell_series_2.to_list()
    return dtw.warping_path(dwell_list_1, dwell_list_2)


def plot_path(path, distance, trial1, trial2, color):
    p1, p2 = zip(*path)
    plt.plot(p1, p2, label=f'Pair {trial1} and {trial2} (Distance: {distance:.2f})', marker='o', color=color)


def save(distance_df: DataFrame, path: str = DTW_Z_V2_CSV):
    distance_df.to_csv(path)
    print("DTW saved to %s" % path)


if __name__ == "__main__":
    dwell_df = read_from_dwell_file()
    analysis_df = read_from_analysis_file()
    distance_df = get_dtw_distance(dwell_df, analysis_df)
    display(distance_df)
