import os
from matplotlib import pyplot as plt
from pandas import DataFrame
import seaborn as sns


def plot_dwell_time_distributions(aoi_analysis_df: DataFrame, aoi: str, to_file: str = None):
    plt.figure(figsize=(10, 6))
    sns.histplot(aoi_analysis_df[aoi], kde=True)
    plt.title('Distribution of %s Dwell Times' % aoi)
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    if to_file is not None:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()
