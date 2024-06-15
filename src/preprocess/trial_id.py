from pandas import DataFrame
from utils.display import display
from utils.paths import TRIAL_INDEX_CSV
from utils.columns import *


def create_trial_id_index(df: DataFrame, to_file: str = None):
    trial_index_df = df[[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]]
    trial_index_df = trial_index_df.drop_duplicates(subset=[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH])
    trial_index_df = trial_index_df.reset_index()
    trial_index_df = trial_index_df.drop(columns=[PID, TRIAL_COUNT])
    trial_index_df[TRIAL_ID] = trial_index_df.index
    trial_index_df = trial_index_df.set_index([SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH])
    if to_file:
        save(trial_index_df, to_file)
    return trial_index_df


def add_trial_id(df: DataFrame, from_index: DataFrame = None):
    trial_index = create_trial_id_index(df) if from_index is None else from_index
    df[TRIAL_ID] = df.loc[df.index][[SELF_LIE, SELF_TRUE, OTHER_LIE, OTHER_TRUTH]].apply(tuple, axis=1).map(lambda x: trial_index.loc[x][TRIAL_ID])
    return df


def save(trial_index_df: DataFrame, path: str = TRIAL_INDEX_CSV):
    trial_index_df.to_csv(path)
    print("Trial Index saved to %s" % path)


