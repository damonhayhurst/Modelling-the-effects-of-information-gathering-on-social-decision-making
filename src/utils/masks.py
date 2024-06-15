from pandas import DataFrame

from utils.columns import *

def get_is_other(df: DataFrame) -> DataFrame:
    """Get bolean value determining whether other box was picked"""
    idx = df.index
    return (df.loc[idx][SELECTION] != df.loc[idx][TRUTH_POSITION]) & (
        df.loc[idx][SELECTION] != df.loc[idx][LIE_POSITION])

def get_is_truth(df: DataFrame) -> DataFrame: 
    idx = df.index
    return df.loc[idx][SELECTION] == df.loc[idx][TRUTH_POSITION]

def get_is_lie(df: DataFrame) -> DataFrame:
    idx = df.index
    return df.loc[idx][SELECTION] == df.loc[idx][LIE_POSITION]

def get_is_lie_or_truth(df: DataFrame) -> DataFrame:
    """Get boolean value whether true or lie box was picked"""
    return get_is_truth(df) | get_is_lie(df)

def get_is_aoi(df: DataFrame, aois: str | list[str] = [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH]) -> DataFrame:
    idx = df.index
    if not isinstance(aois, list):
        aois = [aois]
    return df.loc[idx][AOI].isin(aois)

def is_lie_selected(df: DataFrame, column: str = SELECTED_AOI) -> DataFrame:
    idx = df.index
    return df.loc[idx][column] == LIE

def is_truth_selected(df: DataFrame, column: str = SELECTED_AOI) -> DataFrame:
    idx = df.index
    return df.loc[idx][column] == TRUTH

def is_other_selected(df: DataFrame, column: str = SELECTED_AOI) -> DataFrame:
    idx = df.index
    return df.loc[idx][SELECTED_AOI] == OTHER

def is_selected_aoi_same(df: DataFrame, col1: str = SELECTED_AOI_1, col2: str = SELECTED_AOI_2) -> DataFrame:
    idx = df.index
    return df.loc[idx][col1] == df.loc[idx][col2]

def is_same_pid(df:DataFrame, pid1: str = PID_1, pid2: str = PID_2) -> DataFrame:
    idx = df.index
    return idx.get_level_values(pid1) == idx.get_level_values(pid2)

def is_same_trial(df: DataFrame, trial1: str = TRIAL_1, trial2: str = TRIAL_2) -> DataFrame:
    idx = df.index
    return idx.get_level_values(trial1) == idx.get_level_values(trial2)
