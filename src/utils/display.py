import pandas as pd
import numpy as np
from pandas import DataFrame, MultiIndex
import sys

from utils.columns import MOUSE_TIMESTAMP


def display(df: DataFrame, max_rows: int = 60, max_cols: int = 0, title: str = None):
    """Special PRint function that controls the way Pandas formats the output of a Dataframe"""
    np.set_printoptions(threshold=sys.maxsize)
    with pd.option_context('display.max_rows', max_rows, 'display.max_columns', max_cols):
        if title:
            print("========= %s ==========" % title)
        print(df)
        print('\n')
    np.set_printoptions(threshold=False)
    return df


def display_participant(df: DataFrame, pid: int, max_rows: int = 60, max_cols: int = 0):
    """Print pandas output of a single participant id"""
    display(df.loc[slice(pid, pid), slice(None), slice(None)], max_rows, max_cols)


def display_trials(df: DataFrame, trials: list[(int, int)], max_rows: int = None):
    """Print pandas output for a list of participant trials tuples"""
    trials_df = [df.loc[slice(pid, pid), slice(trial_count, trial_count), slice(None)] for (pid, trial_count) in trials]
    display(pd.concat(trials_df), max_rows)


def display_trial(df: DataFrame, pid: int, trial_count: int, max_rows: int = None):
    """Print pandas output for a single trial from a participant"""
    trial_df = df.loc[slice(pid, pid), slice(trial_count, trial_count), slice(None)]
    display(trial_df, max_rows)


def display_trial_range(df: DataFrame, pid: int, start_trial: int, end_trial: int, max_rows: int = None):
    """Display a range of trials from a participant"""
    trials_df = df.loc[slice(pid, pid), slice(start_trial, end_trial), slice(None)]
    display(trials_df, max_rows)

def display_trial_ids(df: DataFrame):
    idx = df.index
    if MOUSE_TIMESTAMP in idx.names:
        idx: MultiIndex = idx.droplevel(MOUSE_TIMESTAMP)
    trials = idx.unique()
    display(trials.values)
    return df