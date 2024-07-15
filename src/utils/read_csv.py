from typing import Dict, List
from utils.columns import *
from pandas import DataFrame, read_csv, concat
from utils.paths import *
from utils.display import display

INPUT_COLUMNS_IGNORE = [PROLIFIC_ID, GAME, BLOCK, TRIAL_INDEX]
AOI_COLUMNS_IGNORE = [SELF_LIE, SELF_TRUE, OTHER_LIE,  OTHER_TRUTH, RT, YOU_ON_TOP, LIE_POSITION, TRUTH_POSITION, SELECTION, SCREEN_WIDTH]

set_index = DataFrame.set_index
drop = DataFrame.drop

def n_rows(df) -> int:
    """Get number of rows"""
    return len(df)


def column_names(df) -> list[str]:
    """Get column names"""
    return df.keys().values


def print_csv_stats(df):
    """Print statistics related to the input csv"""
    print("\n")
    print("%s rows on csv" % n_rows(df))
    print(", ".join(column_names(df)))
    print("\n")
    return df


def set_data_types_from_csv(df: DataFrame) -> DataFrame:
    """Set data types of columns in data frame from csv"""

    return df.astype({
        PID: int,
        TRIAL_COUNT: int,
        SELF_LIE: int,
        SELF_TRUE: int,
        OTHER_LIE: int,
        OTHER_TRUTH: int,
        RT: int,
        YOU_ON_TOP: int,
        LIE_POSITION: int,
        TRUTH_POSITION: int,
        SELECTION: int,
        SCREEN_WIDTH: int,
    })


def set_index_from_csv(df: DataFrame) -> DataFrame:
    """Set index of the data frame from csv"""
    return df.set_index([PID, TRIAL_COUNT])


def read_from_input_files(paths: list[str] = [YOUNG_ADULTS_1, YOUNG_ADULTS_2]) -> DataFrame:
    csv_dfs = map(lambda path: read_csv(path), paths)
    df = concat(csv_dfs, ignore_index=True)
    df = df.pipe(
        set_data_types_from_csv
    ).pipe(
        set_index, [PID, TRIAL_COUNT]
    ).pipe(
        drop, INPUT_COLUMNS_IGNORE, axis=1
    )
    print_csv_stats(df)
    return df


def read_from_analysis_file(path: str = AOI_ANALYSIS_CSV, index: List[str] = [PID, TRIAL_ID]) -> DataFrame:
    df = read_csv(path)
    df = df.pipe(
        set_data_types_for_analysis_df
    ).pipe(
        set_index, index
    ).pipe(
        dwell_columns_to_seconds, [*DWELL_COLUMNS, AVG_DWELL]
    )
    print("Analysis read from %s \n" % path)
    return df


def set_data_types_for_analysis_df(df: DataFrame) -> DataFrame:
    return df.astype({
        SELECTED_AOI: str,
        FIRST_AOI: str,
        LAST_AOI: str,
        N_AOIS: int,
        UNIQUE_AOIS: int,
        OTHER_TRUTH: "timedelta64[ns]",
        OTHER_LIE: "timedelta64[ns]",
        SELF_TRUE: "timedelta64[ns]",
        SELF_LIE: "timedelta64[ns]",
        AVG_DWELL: "timedelta64[ns]",
        N_ALT_TRANSITIONS: int,
        N_ATT_TRANSITIONS: int,
        PAYNE_INDEX: float,
        TRIAL_ID: int,
        RT: int
    })


def read_from_aois_file(path: str = AOIS_CSV) -> DataFrame:
    df = read_csv(path)
    df = df.pipe(
        drop, AOI_COLUMNS_IGNORE, axis=1
    ).pipe(
        set_data_types_for_aois_df
    ).pipe(
        set_index, [PID, TRIAL_COUNT, MOUSE_TIMESTAMP]
    )
    print("AOIs read from %s \n" % path)
    return df


def set_data_types_for_aois_df(df: DataFrame) -> DataFrame:
    return df.astype({
        PID: int,
        TRIAL_COUNT: int,
        MOUSE_X: int,
        MOUSE_Y: int,
        SELECTED_AOI: str,
        AOI: str,
        MOUSE_TIMESTAMP: "datetime64[ns]"
    })


def read_from_dwell_file(path: str = DWELL_TIMELINE_CSV) -> DataFrame:
    df = read_csv(path)
    df = df.pipe(
        set_data_types_for_dwell_df
    ).pipe(
        set_index, [PID, TRIAL_ID, MOUSE_TIMESTAMP]
    ).pipe(
        dwell_columns_to_seconds, [DWELL_TIME]
    )
    print("Dwell Timeline from %s \n" % path)
    return df


def set_data_types_for_dwell_df(df: DataFrame) -> DataFrame:
    return df.astype({
        PID: int,
        TRIAL_COUNT: int,
        DWELL_TIME: "timedelta64[ns]",
        MOUSE_TIMESTAMP: "datetime64[ns]"
    })


def dwell_columns_to_seconds(df: DataFrame, cols: list[str]):
    for column in cols:
        df[column] = df[column].apply(lambda timedelta: timedelta.total_seconds())
    return df

def read_from_dtw_file(path: str = DTW_CSV) -> DataFrame:
    df = read_csv(path)
    df = df.pipe(
        set_data_types_for_dtw_file
    ).pipe(
        set_index, [PID_1, TRIAL_ID_1, TRIAL_COUNT_1, PID_2, TRIAL_ID_2, TRIAL_COUNT_2]
    )
    print("DTW read from %s \n" % path)
    return df

def set_data_types_for_dtw_file(df: DataFrame) -> DataFrame:
    return df.astype({
        PID_1: int,
        TRIAL_ID_1: int,
        PID_2: int,
        TRIAL_ID_2: int,
        DISTANCE: float
    })

def read_from_trial_index_file(path: str = TRIAL_INDEX_CSV) -> DataFrame:
    df = read_csv(path)
    df = df.pipe(
        set_data_types_for_trial_index_file
    ).pipe(
        set_index, [TRIAL_ID]
    )
    print("Trial Index read from %s \n" % path)
    return df

def set_data_types_for_trial_index_file(df: DataFrame) -> DataFrame:
    return df.astype({
        SELF_LIE: int, 
        SELF_TRUE: int, 
        OTHER_LIE: int, 
        OTHER_TRUTH: int, 
        TRIAL_ID: int
    })


def set_data_types_for_cluster_file(df: DataFrame) -> DataFrame:
    return df.astype({
        PID: int,
        TRIAL_ID: int,
        CLUSTER: int
    })

def read_from_cluster_file(path: str = TIME_SERIES_KMEANS_2_CLUSTER_CSV) -> DataFrame:
    df = read_csv(path)
    df = df.pipe(
        set_data_types_for_cluster_file
    ).pipe(
        set_index, [PID, TRIAL_ID]
    )
    print('Cluster file read from %s \n' % path)
    return df

def read_friom_cluster_files_to_dict(paths: List[str]) -> Dict[int, DataFrame]:
    cluster_dict = {}
    for path in paths:
        df = read_from_cluster_file(path)
        n_cluster = df[CLUSTER].max()
        if n_cluster in cluster_dict:
            print('Error: Cluster file already exists for n_cluster %s' % n_cluster)
        else:
            cluster_dict[n_cluster] = df
    return cluster_dict

