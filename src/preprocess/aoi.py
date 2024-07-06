from pandas import DataFrame, MultiIndex, to_datetime
from utils.columns import *
from utils.display import display
from utils.masks import get_is_lie, get_is_lie_or_truth, get_is_other, get_is_truth
from utils.paths import AOIS_CSV
from preprocess.filtering import do_filtering, get_n_trials, print_n_participants
from utils.read_csv import read_from_input_files
from preprocess.trial_id import add_trial_id


def add_selected_aoi(df: DataFrame, bypass: bool = False):
    """Add selected box associated with either TRUTH or LIE or OTHER aoi"""
    if bypass:
        return df
    is_lie, is_truth, is_other = get_is_lie(df), get_is_truth(df), get_is_other(df)
    df[SELECTED_AOI] = None
    df.loc[is_lie, SELECTED_AOI] = LIE
    df.loc[is_truth, SELECTED_AOI] = TRUTH
    df.loc[is_other, SELECTED_AOI] = OTHER
    return df


def separate_mouse_coords(df: DataFrame) -> DataFrame:
    """Transform data frame from csv to one that has individual rows for each recorded mouse coordinate.
        Any mouse coordinates that are malformatted correctly or lacking any mouse coordinate. Have their
        trial removed."""
    coords_arr_df = df[MOUSE_COORD].str.split(';')
    coord_row_df = coords_arr_df.explode()
    split_coord_row_df = coord_row_df.str.split(",", expand=True)
    valid_coord_col_limit = 2
    invalid_cols = [col for col in split_coord_row_df if int(
        col) > valid_coord_col_limit]
    valid_cols = [col for col in split_coord_row_df if int(
        col) <= valid_coord_col_limit]
    invalid_cols_df = split_coord_row_df.loc[:, invalid_cols]
    valid_cols_df = split_coord_row_df.loc[:, valid_cols]
    has_val_in_invalid = invalid_cols_df.notna().all(axis=1) if not invalid_cols_df.empty else []
    has_no_val_in_valid = valid_cols_df.isna().any(axis=1)
    no_ts_df = split_coord_row_df.loc[has_no_val_in_valid]
    print("%s trials with no timestamps" % len(no_ts_df.index))
    malformatted_df = split_coord_row_df.loc[has_val_in_invalid]
    print("%s trials with malformatted timestamps" % len(malformatted_df.index))
    valid_coord_row_df = split_coord_row_df.drop(labels=no_ts_df.index).drop(labels=malformatted_df.index).drop(invalid_cols, axis=1)
    valid_coord_row_df.rename(columns={0: MOUSE_X, 1: MOUSE_Y, 2: MOUSE_TIMESTAMP}, inplace=True)
    valid_coord_row_df[MOUSE_X], valid_coord_row_df[MOUSE_Y] = valid_coord_row_df[MOUSE_X].astype(int), valid_coord_row_df[MOUSE_Y].astype(int)
    valid_coord_row_df[MOUSE_TIMESTAMP] = to_datetime(valid_coord_row_df[MOUSE_TIMESTAMP].astype(int), unit="ms", errors="coerce")
    invalid_ts = valid_coord_row_df[MOUSE_TIMESTAMP][valid_coord_row_df[MOUSE_TIMESTAMP].isna()]
    valid_coord_row_df.drop(labels=invalid_ts.index)
    transformed_df = valid_coord_row_df.join(df)
    transformed_df = transformed_df.set_index(MOUSE_TIMESTAMP, append=True)
    transformed_df = transformed_df.drop([MOUSE_COORD], axis=1)
    return transformed_df


def get_is_screen_width_zero(df: DataFrame):
    return df[SCREEN_WIDTH] == 0


def remove_screen_width_zero(df: DataFrame):
    is_screen_width_zero = get_is_screen_width_zero(df)
    zeroes = df.loc[is_screen_width_zero]
    df = df.drop(labels=zeroes.index)
    print("%s trials with no screen width" % get_n_trials(zeroes))
    return df


def get_box_x_fn(screen_width):
    x_mid = screen_width/2
    boxes_length = 1193
    x_start = x_mid - (boxes_length/2)
    box, gap = 200, 131

    def get_box_x(n):
        box_x = [
            (x_start, x_start + box),
            (x_start + box + gap, x_start + box + gap + box),
            (x_start + box + gap + box + gap, x_start + box + gap + box + gap + box),
            (x_start + box + gap + box + gap + box + gap, x_start + box + gap + box + gap + box + gap + box)
        ]
        return box_x[n-1]
    return get_box_x


def get_all_box_x(screen_width):
    get_box_x = get_box_x_fn(screen_width)
    box_x = [
        get_box_x(1), get_box_x(2), get_box_x(3), get_box_x(4)
    ]
    return box_x


def get_is_box_x_fn(screen_width):
    def is_box(n, x):
        get_box_x = get_box_x_fn(screen_width)
        is_box = [
            lambda x: (x >= get_box_x(1)[0]) & (x <= get_box_x(1)[1]),
            lambda x: (x >= get_box_x(2)[0]) & (x <= get_box_x(2)[1]),
            lambda x: (x >= get_box_x(3)[0]) & (x <= get_box_x(3)[1]),
            lambda x: (x >= get_box_x(4)[0]) & (x <= get_box_x(4)[1]),
        ]
        return is_box[n-1](x)
    return is_box


def is_top_choice(y): return (y >= 314) & (y <= 404)
def is_bottom_choice(y): return (y >= 439) & (y <= 529)


def is_area_of_interest(row):
    screen_width = row[SCREEN_WIDTH]
    if screen_width == 0:
        return None
    x, y = row[MOUSE_X], row[MOUSE_Y]
    is_box = get_is_box_x_fn(screen_width)
    for box_n in range(1, 5):
        if is_box(box_n, x):
            if row[TRUTH_POSITION] == box_n:
                if (is_top_choice(y) and row[YOU_ON_TOP]) or (is_bottom_choice(y) and not row[YOU_ON_TOP]):
                    return SELF_TRUE
                else:
                    return OTHER_TRUTH
            elif row[LIE_POSITION] == box_n:
                if (is_top_choice(y) and row[YOU_ON_TOP]) or (is_bottom_choice(y) and not row[YOU_ON_TOP]):
                    return SELF_LIE
                else:
                    return OTHER_LIE
            else:
                return None
    return None


def print_n_trials(df: DataFrame):
    print("Trials: %s" % get_n_trials(df))


def get_trials_has_no_aois(df: DataFrame) -> DataFrame:
    df_by_trial = df.groupby(level=[PID, TRIAL_COUNT])
    has_no_aois = df_by_trial[AOI].apply(lambda aoi: aoi.isnull().all())
    return df.loc[has_no_aois]


def get_trials_has_aois(df: DataFrame) -> DataFrame:
    df_by_trial = df.groupby(level=[PID, TRIAL_COUNT])
    has_aois = df_by_trial[AOI].apply(lambda aoi: aoi.notnull().any())
    return df.loc[has_aois]


def remove_trials_with_no_aois(df: DataFrame) -> DataFrame:
    no_aois = get_trials_has_no_aois(df)
    aois_df = df.drop(labels=no_aois.index)
    return aois_df


def remove_pid_with_incomplete_n_trials(df: DataFrame, n_trials: int = 80, bypass: bool = False) -> DataFrame:
    if not bypass:
        counts = df.groupby(PID).apply(lambda x: x.index.get_level_values(TRIAL_COUNT).nunique())
        valid_pid = counts[counts >= n_trials].index
        invalid_pid = counts[counts < n_trials].index
        print("Less than %s trials Participants Removed: %s" % (n_trials, len(invalid_pid)))
        return df[df.index.get_level_values(PID).isin(valid_pid)]
    else:
        return df

    

def print_aois_stats(df: DataFrame):
    no_aois = get_trials_has_no_aois(df)
    print("%s trials with no AOIs" % get_n_trials(no_aois))
    is_lie_or_truth = get_is_lie_or_truth(no_aois)
    no_aois_with_truth_or_lie: DataFrame = no_aois.loc[is_lie_or_truth]
    print("%s trials with no AOIs that involve lie or truth selection" % get_n_trials(no_aois_with_truth_or_lie))
    has_aois = get_trials_has_aois(df)
    print("Trials: %s" % get_n_trials(has_aois))
    print('\n')
    return df


def determine_aoi(df: DataFrame, to_file: str = None) -> DataFrame:
    df = add_selected_aoi(df)
    coord_row_df = separate_mouse_coords(df)
    coord_row_df = remove_screen_width_zero(coord_row_df)
    coord_row_df[AOI] = coord_row_df.apply(is_area_of_interest, axis=1)
    print_n_trials(coord_row_df)
    print("\n")
    print_aois_stats(coord_row_df)
    aoi_df = remove_trials_with_no_aois(coord_row_df)
    aoi_df = remove_pid_with_incomplete_n_trials(aoi_df, 80, bypass=True)
    if to_file:
        save(aoi_df, to_file)
    return aoi_df


def save(aoi_df: DataFrame, path: str = AOIS_CSV):
    aoi_df.to_csv(path, na_rep='None')
    print("AOIs saved to %s" % path)

