from functools import partial
from typing import Callable, Literal
from pandas import DataFrame, Series, Timedelta, notna
from utils.columns import *
from utils.display import *
from preprocess.dwell import get_dwell_spans_for_aoi
from utils.paths import AOI_ANALYSIS_CSV, AVERAGE_ANALYSIS_CSV
from utils.read_csv import read_from_aois_file


def get_first_aoi(aoi_df: DataFrame) -> DataFrame:
    """Get first AOI in each trial.

    Args:
        df (DataFrame): AOI timeline

    Returns:
        DataFrame: Dataframe representing first AOI per trial
    """
    trial_df = aoi_df.groupby(level=[PID, TRIAL_COUNT])
    return trial_df.agg({AOI: lambda trial: trial.dropna().iloc[0]})


def get_last_aoi(df: DataFrame) -> DataFrame:
    """Get last AOI in each trial.

    Args:
        df (DataFrame): AOI timeline

    Returns:
        DataFrame: Dataframe representing last AOI per trial
    """
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])
    return trial_df.agg({AOI: lambda trial: trial.dropna().iloc[-1]})


def get_n_aois_per_trial(df: DataFrame) -> DataFrame:
    """Get number of aoi occurences in each trial.

    Args:
        df (DataFrame): AOI timeline

    Returns:
        DataFrame: Dataframe representing number of AOIs per trial
    """
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])

    def n_aois(trial: Series):
        current, n = None, 0
        for aoi in trial:
            if aoi != current and aoi:
                n += 1
            current = aoi
        return n
    return trial_df.agg({AOI: n_aois})


def get_unique_aois(df: DataFrame) -> Series:
    """Get unique AOIs per trial.

    Args:
        df (DataFrame): AOI timeline

    Returns:
        DataFrame: Dataframe representing last AOI per trial
    """
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])
    return trial_df[AOI].apply(lambda aoi: aoi.drop_duplicates().count())


def get_avg_dwell_time(df: DataFrame, aoi: str) -> Series:
    """Get average dwell time for a given AOI per trial.

    Args:
        df (DataFrame): AOI timeline
        aoi (str): one of four AOIs: SELF TRUE, SELF LIE, OTHER LIE, OTHER TRUTH

    Returns:
        Series: average dwell time per trial for given AOI
    """
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])

    def avg_dwell_time(df: DataFrame, aoi) -> Timedelta:
        timespans = get_dwell_spans_for_aoi(df, aoi)
        avg_dwell_for_aoi = timespans[DWELL_TIME].mean() if notna(timespans[DWELL_TIME].mean()) else Timedelta(0)
        return avg_dwell_for_aoi

    return trial_df.apply(avg_dwell_time, aoi)


def n_transitions(trial: Series, is_aoi: Callable[[str], bool], is_other_aoi: Callable[[str], bool]) -> int:
    """ Number of transitions between two boolean conditional checks for aoi.

    Args:
        trial (Series): timeline of AOIs for a trial
        is_aoi (Callable[[str], bool]): conditional check for AOI
        is_other_aoi (Callable[[str], bool]): conditional check for other AOI

    Returns:
        int: number of transitions between conditional checks
    """
    current, n = None, 0
    for aoi in trial:
        if not current:
            if is_aoi(aoi):
                current = is_aoi
            if is_other_aoi(aoi):
                current = is_other_aoi
        else:
            if current is is_aoi and is_other_aoi(aoi):
                n += 1
                current = is_other_aoi
            elif current is is_other_aoi and is_aoi(aoi):
                n += 1
                current = is_aoi
    return n


def get_n_att_transitions(df: DataFrame) -> DataFrame:
    """Number of attribute AOI transitionas. Attribute transitions defined as between SELF and OTHER AOIs.

    Args:
        df (DataFrame): AOI timeline

    Returns:
        DataFrame: Dataframe representing number of transitions between attributional AOIs
    """
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])
    def is_self(aoi): return aoi == SELF_LIE or aoi == SELF_TRUE
    def is_other(aoi): return aoi == OTHER_LIE or aoi == OTHER_TRUTH
    n_alt_transitions = partial(n_transitions, is_aoi=is_self, is_other_aoi=is_other)

    return trial_df[AOI].agg(n_transitions=n_alt_transitions)


def get_n_alt_transitions(df: DataFrame) -> DataFrame:
    """Number of alternate AOI transitionas. Alternate transitions defined as between TRUTH and LIE AOIs.

    Args:
        df (DataFrame): AOI timeline

    Returns:
        DataFrame: Dataframe representing number of transitions between alternating AOIs
    """
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])
    def is_lie(aoi): return aoi == SELF_LIE or aoi == OTHER_LIE
    def is_truth(aoi): return aoi == SELF_TRUE or aoi == OTHER_TRUTH
    n_alt_transitions = partial(n_transitions, is_aoi=is_lie, is_other_aoi=is_truth)

    return trial_df[AOI].agg(n_transitions=n_alt_transitions)


def create_aoi_analysis(aoi_df: DataFrame, to_file: str = None) -> DataFrame:
    """Create analysis DataFrame per trial based on an analysis of the aoi timeline for each given trial.

    Args:
        aoi_df (DataFrame): AOI Timeline of all trials
        to_file (str, optional): File path to save analysis to. Defaults to None.

    Returns:
        DataFrame: Dataframe representing analysis of AOIs per trial.
    """
    analysis_df = DataFrame()
    analysis_df[TRIAL_ID] = aoi_df[TRIAL_ID]
    analysis_df[SELECTED_AOI] = aoi_df[SELECTED_AOI]
    analysis_df[FIRST_AOI] = get_first_aoi(aoi_df)
    analysis_df[LAST_AOI] = get_last_aoi(aoi_df)
    analysis_df[N_AOIS] = get_n_aois_per_trial(aoi_df)
    analysis_df[UNIQUE_AOIS] = get_unique_aois(aoi_df)
    analysis_df[OTHER_TRUTH] = get_avg_dwell_time(aoi_df, OTHER_TRUTH)
    analysis_df[OTHER_LIE] = get_avg_dwell_time(aoi_df, OTHER_LIE)
    analysis_df[SELF_LIE] = get_avg_dwell_time(aoi_df, SELF_LIE)
    analysis_df[SELF_TRUE] = get_avg_dwell_time(aoi_df, SELF_TRUE)
    analysis_df[AVG_DWELL] = analysis_df[[OTHER_TRUTH, OTHER_LIE, SELF_LIE, SELF_TRUE]].mean(axis=1)
    n_att_transitions = get_n_att_transitions(aoi_df)
    n_alt_transitions = get_n_alt_transitions(aoi_df)
    analysis_df[N_TRANSITIONS] = n_att_transitions + n_alt_transitions
    analysis_df[N_ATT_TRANSITIONS] = n_att_transitions
    analysis_df[N_ALT_TRANSITIONS] = n_alt_transitions
    payne_index = ((n_alt_transitions - n_att_transitions) / (n_alt_transitions + n_att_transitions))
    analysis_df[PAYNE_INDEX] = payne_index.fillna(0)
    analysis_df[RT] = aoi_df[RT]
    analysis_df = analysis_df.reset_index()
    analysis_df = analysis_df.drop_duplicates(subset=[PID, TRIAL_COUNT])
    analysis_df = analysis_df.drop(columns=MOUSE_TIMESTAMP)
    analysis_df = analysis_df.set_index([PID, TRIAL_ID])
    if to_file:
        save_analysis(analysis_df, to_file)
    return analysis_df

def save_analysis(analysis_df: DataFrame, path: str = AOI_ANALYSIS_CSV):
    """Save analysis to file with success message.

    Args:
        analysis_df (DataFrame): Dataframe representing analysis per trial.
        path (str, optional): Path to save to.
    """
    analysis_df.to_csv(path, na_rep='None')
    print("Analysis saved to %s" % path)

def create_average_analysis(analysis_df: DataFrame, to_file: str = None):
    """Create average analysis per participant.

    Args:
        analysis_df (DataFrame): AOI Analysis
        to_file (str, optional): Path to save to. Defaults to None.
    """

    def aoi_mode(series: Series):
        modes = series.mode()
        if not modes.empty:
            return " ".join(modes)
        return None

    average_df = analysis_df.groupby(PID).agg({
        SELECTED_AOI: aoi_mode,
        N_AOIS: "mean",
        UNIQUE_AOIS: "mean",
        OTHER_TRUTH: "mean",
        OTHER_LIE: "mean",
        SELF_LIE: "mean",
        SELF_TRUE: "mean",
        N_ALT_TRANSITIONS: "mean",
        N_ATT_TRANSITIONS: "mean"
    })

    average_df = average_df.round({
        N_AOIS: 0,
        UNIQUE_AOIS: 0,
        N_ALT_TRANSITIONS: 0,
        N_ATT_TRANSITIONS: 0
    })

    average_df = average_df.astype({
        N_AOIS: int,
        UNIQUE_AOIS: int,
        N_ALT_TRANSITIONS: int,
        N_ATT_TRANSITIONS: int
    })

    if to_file:
        save_average_analysis(average_df, to_file)

    return average_df


def save_average_analysis(average_df: DataFrame, path: str = AVERAGE_ANALYSIS_CSV):
    """Save per paraticpant average analysis of AOIs.

    Args:
        average_df (DataFrame): Analysis of AOIs per trial
        path (str, optional): Path to save to
    """
    average_df.to_csv(path)
    print("Average analysis saved to %s" % path)


