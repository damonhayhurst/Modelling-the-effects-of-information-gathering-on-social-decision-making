from pandas import DataFrame, merge
from kmeans.clustering import do_clustering, get_pca_col, plot_scatter
from utils.columns import *
from utils.paths import *
from utils.display import display
from utils.masks import is_lie_selected, is_truth_selected, is_other_selected
from utils.read_csv import read_from_analysis_file


def map_columns_to_selected_aoi_masks(df: DataFrame) -> DataFrame:
    df[IS_LIE] = is_lie_selected(df, SELECTED_AOI)
    df[IS_TRUTH] = is_truth_selected(df, SELECTED_AOI)
    df[IS_OTHER] = is_other_selected(df, SELECTED_AOI)
    return df


def merge_analysis(aoi_df: DataFrame, cluster_df: DataFrame):
    cluster_df = cluster_df.reset_index()
    aoi_df = aoi_df.reset_index()
    df = merge(aoi_df[[SELECTED_AOI]], cluster_df, left_index=True, right_index=True, how="left")
    df = map_columns_to_selected_aoi_masks(df)
    df = df.drop(columns="index")
    return df


def cluster_analysis(df: DataFrame):
    by_cluster = df.groupby(CLUSTER)
    return by_cluster.agg({
        IS_LIE: [("Total lies", "sum"), ("Mean lies", "mean")],
        IS_TRUTH: [("Total truth", "sum"), ("Mean trues", "mean")],
        IS_OTHER: [("Total other", "sum"), ("Mean other", "mean")],
    })


def pca_analysis(df: DataFrame):
    by_cluster = df.groupby(CLUSTER)
    pca_columns = [col for col in df.columns if col.startswith('Principal Component')]
    pca_summ = {col: ['mean'] for col in pca_columns}
    return by_cluster.agg(pca_summ)


def filter_selected(df: DataFrame, aoi: str):
    if aoi == LIE:
        return df[is_lie_selected(df)]
    if aoi == TRUTH:
        return df[is_truth_selected(df)]
    if aoi == OTHER:
        return df[is_other_selected(df)]
    else:
        return df


if __name__ == '__main__':
    aoi_df = read_from_analysis_file(AOI_ANALYSIS_CSV)
    cluster_df = do_clustering(aoi_df, n_components=0.5, columns=[*DWELL_COLUMNS, PAYNE_INDEX])
    analysis_df = merge_analysis(aoi_df, cluster_df).pipe(display)
    cluster_analysis_df = cluster_analysis(analysis_df).pipe(display)
    plot_scatter(analysis_df, x=get_pca_col(1), y=get_pca_col(2))
