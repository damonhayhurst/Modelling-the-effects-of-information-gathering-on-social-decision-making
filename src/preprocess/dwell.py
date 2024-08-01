from datetime import timedelta
from pandas import Index
from utils.columns import *
from utils.display import *
from utils.masks import get_is_aoi
from utils.read_csv import *


def get_dwell_spans_for_aoi(df: DataFrame, aoi):
    is_aoi = get_is_aoi(df, aoi)
    is_toggled = is_aoi != is_aoi.shift(1, fill_value=False)
    range_groups = is_toggled.cumsum()
    aoi_range_groups = range_groups[is_aoi].reset_index()
    timespan_groups = aoi_range_groups.groupby(by=AOI)
    timespans = timespan_groups[MOUSE_TIMESTAMP].agg(["min", "max"])
    timespans[DWELL_TIME] = timespans["max"] - timespans["min"]
    return timespans


def get_all_dwell_spans(df: DataFrame):
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])
    dwell_by_aoi_dfs = []
    for aoi in [SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH, None]:
        dwell_df: DataFrame = trial_df.apply(get_dwell_spans_for_aoi, aoi)
        dwell_df = dwell_df.droplevel(AOI)
        dwell_df[AOI] = aoi
        dwell_by_aoi_dfs.append(dwell_df)
    return concat(dwell_by_aoi_dfs).sort_values(by="min")


def create_dwell_timeline(df: DataFrame, to_file: str = None):
    dwell_df = get_all_dwell_spans(df)
    dwell_df[MOUSE_TIMESTAMP] = dwell_df["min"]
    dwell_df = dwell_df.drop(columns=["min", "max"])
    dwell_df = add_trial_id(dwell_df, df)
    dwell_df.set_index([dwell_df.index, MOUSE_TIMESTAMP])
    dwell_df = dwell_df.sort_index()
    # dwell_df = smoothing(dwell_df)
    if to_file:
        save(dwell_df, to_file)
    return dwell_df



# def smoothing(dwell_df, less_than=200):
#     less_than_duration = dwell_df[DWELL_TIME] < timedelta(milliseconds=less_than)
#     display(dwell_df[less_than_duration])
#     return dwell_df.drop(index=less_than_duration)


def add_trial_id(dwell_df: DataFrame, aoi_df: DataFrame):
    def get_trial_id(row):
        return aoi_df.loc[(row.name[0], row.name[1])][TRIAL_ID].values[0]

    dwell_df[TRIAL_ID] = dwell_df.apply(get_trial_id, axis=1)
    return dwell_df


def save(dwell_df: DataFrame, path: str = DWELL_TIMELINE_CSV):
    dwell_df.to_csv(path, na_rep='None')
    print("Dwell Timeline saved to %s" % path)


def get_start_and_end_of_trial(df: DataFrame):
    df = df.reset_index(MOUSE_TIMESTAMP)
    trial_df = df.groupby(level=[PID, TRIAL_COUNT])
    start, end = trial_df.first()[MOUSE_TIMESTAMP], trial_df.last()[MOUSE_TIMESTAMP]
    return start, end
