from pandas import DataFrame

from utils.columns import *
from utils.columns import OTHER_LOSS, SELF_GAIN


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


def is_same_pid(df: DataFrame, pid1: str = PID_1, pid2: str = PID_2) -> DataFrame:
    idx = df.index
    return idx.get_level_values(pid1) == idx.get_level_values(pid2)


def is_same_trial(df: DataFrame, trial1: str = TRIAL_1, trial2: str = TRIAL_2) -> DataFrame:
    idx = df.index
    return idx.get_level_values(trial1) == idx.get_level_values(trial2)


def is_no_dwell_for_aois(aoi_analysis_df: DataFrame, aois: list[str] = [SELF_TRUE, SELF_LIE, OTHER_TRUTH, OTHER_LIE]) -> DataFrame:
    return (aoi_analysis_df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]] == 0).any(axis=1)


def get_positive_gain(gains_df: DataFrame):
    return gains_df[SELF_GAIN] > 0


def get_gain_of_ten(gains_df: DataFrame):
    return gains_df[SELF_GAIN] > 9

def get_gain_of_between_ten_and_twenty(gains_df: DataFrame):
    return (gains_df[SELF_GAIN] > 9) & (gains_df[SELF_GAIN] < 21)

def get_gain_of_twenty(gains_df: DataFrame):
    return gains_df[SELF_GAIN] > 19


def get_gain_of_between_twenty_and_thirty(gains_df: DataFrame):
    return (gains_df[SELF_GAIN] > 19) & (gains_df[SELF_GAIN] < 31)


def get_gain_of_thirty(gains_df: DataFrame):
    return gains_df[SELF_GAIN] > 29



def get_positive_gain_of_less_than_ten(gains_df: DataFrame):
    return (gains_df[SELF_GAIN] <= 9) & (gains_df[SELF_GAIN] > 0)


def get_gain_of_less_than_ten(gains_df: DataFrame):
    return gains_df[SELF_GAIN] <= 9


def get_negative_gain(gains_df: DataFrame):
    return gains_df[SELF_GAIN] < 0


def get_loss_of_ten(gains_df: DataFrame):
    return gains_df[OTHER_LOSS] > 9


def get_loss_of_less_than_ten(gains_df: DataFrame):
    return (gains_df[OTHER_LOSS] <= 9) & (gains_df[OTHER_LOSS] > 0)


def get_negative_loss(gains_df: DataFrame):
    return gains_df[OTHER_LOSS] < 0


def get_loss_of_twenty(gains_df: DataFrame):
    return gains_df[OTHER_LOSS] > 19


def get_loss_of_thirty(gains_df: DataFrame):
    return gains_df[OTHER_LOSS] > 29
