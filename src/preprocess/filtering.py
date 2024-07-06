from utils.columns import *
from pandas import DataFrame, Series
from utils.masks import get_is_lie, get_is_other, get_is_truth
from utils.read_csv import read_from_input_files


def print_n_participants(df: DataFrame) -> DataFrame:
    """Print number of participants in data frame"""
    print("Participants: %s" % len(get_participants(df)))
    return df


def get_participants(df: DataFrame):
    """Get all participant ids from a data frame"""
    return df.index.get_level_values(PID).unique()


def remove_practices(df: DataFrame, bypass: bool = False) -> DataFrame:
    """Remove practices from a data set. Practice trials are evaluated as the first 4 trials of a participant.
        Trial count after that returns to one, at which the real trials begin."""
    if bypass:
        return df

    def remove_practices(trials: Series):
        practices = []
        for i in range(0, len(trials) + 1):
            trial_count = trials.iloc[i].name[1]
            if trial_count in practices:
                return trials.iloc[i:]
            else:
                practices.append(trial_count)

    by_pid = df.groupby(PID)
    filtered: DataFrame = by_pid.apply(remove_practices)
    filtered = filtered.reset_index(level=1, drop=True)
    return filtered


def get_n_trials_by_pid(df: DataFrame) -> DataFrame:
    """Get number of trials per participant id"""
    return df.groupby(level=PID).size()


def get_n_trials(df: DataFrame):
    return len(df.groupby(level=[PID, TRIAL_COUNT]))


def print_n_trials(df: DataFrame):
    print("Trials: %s" % get_n_trials(df))


def remove_dnf(df: DataFrame, bypass: bool = False) -> DataFrame:
    """Remove participants who did not finish the experiement. The number of trials determined to be in a finished
        state is 80."""
    if bypass:
        return df
    n_trials = get_n_trials_by_pid(df)
    dnfs = n_trials[n_trials < 80]
    filtered_df = df.drop(labels=dnfs.index)
    print("DNF Participants Removed: %s" % len(get_participants(dnfs)))
    return filtered_df


def get_n_lies(df: DataFrame) -> DataFrame:
    """Get number of lies per participant"""
    is_lie = get_is_lie(df)
    n_lies = is_lie.groupby(level=PID).sum()
    return n_lies


def get_percent_lies(df: DataFrame):
    """Get percentage lies over the total trials for a participant"""
    n_trials = get_n_trials_by_pid(df)
    n_lies = get_n_lies(df)
    lie_ratio = (n_lies / n_trials)
    lie_percent = lie_ratio * 100
    return lie_percent


def remove_percent_lies(df: DataFrame, percent: int = 95, bypass: bool = False):
    """Remove participants who lied on percent of trials"""
    if bypass:
        return df
    percent_lies = get_percent_lies(df)
    over_95_percent_lies = percent_lies[percent_lies > percent]
    filtered_df = df.drop(labels=over_95_percent_lies.index)
    print("Over %s%% Lies Participants Removed: %s" %
          (percent, len(get_participants(over_95_percent_lies))))
    return filtered_df


def get_n_truths(df: DataFrame):
    """Get number of truths per participant"""
    is_truth = get_is_truth(df)
    n_truths = is_truth.groupby(level=PID).sum()
    return n_truths


def get_percent_truths(df: DataFrame):
    """Get percentage of truths over the total trials of each participant"""
    n_trials = get_n_trials_by_pid(df)
    n_truths = get_n_truths(df)
    truth_ratio = (n_truths / n_trials)
    truth_percent = truth_ratio * 100
    return truth_percent


def remove_percent_truths(df: DataFrame, percent: int = 95, bypass: bool = False):
    """Remove participants who told the truth on over a percent of trials"""
    if bypass:
        return df
    percent_truths = get_percent_truths(df)
    over_95_percent_truths = percent_truths[percent_truths > percent]
    filtered_df = df.drop(labels=over_95_percent_truths.index)
    print("Over %s%% Truths Participants Removed: %s" %
          (percent, len(get_participants(over_95_percent_truths))))
    return filtered_df


def get_n_others(df: DataFrame):
    """Get number of times participants picked another box that wasn't lie or truth (blank) per participant"""
    is_other = get_is_other(df)
    n_others = is_other.groupby(level=PID).sum()
    return n_others


def get_percent_others(df: DataFrame):
    """Get percentage of trials where other box was picked per participant"""
    n_trials = get_n_trials_by_pid(df)
    n_others = get_n_others(df)
    other_ratio = (n_others / n_trials)
    other_percent = other_ratio * 100
    return other_percent


def remove_percent_others(df: DataFrame, percent: int = 5, bypass: bool = False):
    """Remove participants who picked the other box on over 5% of their trials"""
    if bypass:
        return df
    percent_others = get_percent_others(df)
    over_percent_others = percent_others[percent_others > percent]
    filtered_df = df.drop(labels=over_percent_others.index)
    print("Over %s%% Other Boxes Participants Removed: %s" %
          (percent, len(get_participants(over_percent_others))))
    return filtered_df


def remove_no_mouse_coords(df: DataFrame, bypass: bool = False):
    """Remove participants who have no mouse coordinates associated with any of their recorded trial attempts"""
    if bypass:
        return df
    has_no_mouse_coords = df.groupby(level=PID)[MOUSE_COORD].apply(
        lambda trial: trial.isna().all())
    no_mouse_coords = has_no_mouse_coords[has_no_mouse_coords]
    filtered_df = df.drop(labels=no_mouse_coords.index)
    print("No Mouse Coords Participants Removed: %s" %
          len(get_participants(no_mouse_coords)))
    return filtered_df


def remove_over_percent_no_mouse_coords(df: DataFrame, percent: int = 95, bypass: bool = False):
    """Remove participants who have more than some percent of no mouse coordinates"""
    if bypass:
        return df
    proportion = percent / 100
    has_percent_no_mouse_coords = df.groupby(level=PID)[MOUSE_COORD].apply(
        lambda trial: trial.isna().mean())
    no_mouse_coords = has_percent_no_mouse_coords[has_percent_no_mouse_coords > proportion]
    filtered_df = df.drop(labels=no_mouse_coords.index)
    print("Over %s%% No Mouse Coords Participants Removed: %s" %
          (percent, len(get_participants(no_mouse_coords))))
    return filtered_df


def print_stats(df: DataFrame):
    print_n_participants(df)
    print_n_trials(df)
    print('\n')


def do_filtering(df: DataFrame) -> DataFrame:
    """Pipeline that removes participants based on a number of factors from the dataframe provided by the csv"""
    df = df.pipe(
        remove_practices, bypass=False
    ).pipe(
        remove_dnf
    ).pipe(
        remove_percent_lies, percent=95
    ).pipe(
        remove_percent_truths, percent=95
    ).pipe(
        remove_percent_others, percent=5
    ).pipe(
        remove_no_mouse_coords, bypass=True
    ).pipe(
        remove_over_percent_no_mouse_coords, percent=85
    )
    return df
