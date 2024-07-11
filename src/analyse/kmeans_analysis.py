
import os
from typing import List
from matplotlib import pyplot as plt
from numpy import linspace
import pandas as pd
from pandas import Categorical, DataFrame
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns
from utils.columns import CLUSTER, IS_LIE, IS_OTHER, IS_TRUTH, N_CLUSTER, OTHER_LIE, OTHER_TRUTH, PAYNE_INDEX, PID, SELECTED_AOI, SELF_LIE, SELF_TRUE, SILHOUETTE, TRIAL_COUNT, TRIAL_ID
from utils.display import display
from utils.masks import is_lie_selected, is_other_selected, is_truth_selected


def get_pca_col(i): return "Principal Component %s" % i
def get_pca_cols(n): return [get_pca_col(i) for i in range(1, n + 1)]


def pca(aoi_df: DataFrame, n_components=0.9) -> DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(aoi_df)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    columns = get_pca_cols(pca.n_components_)
    principal_df = DataFrame(data=principal_components, columns=columns)
    print_pca_stats(aoi_df, pca)
    aoi_df = aoi_df.reset_index()
    df = pd.merge(aoi_df, principal_df, left_index=True, right_index=True, how="left")
    df.set_index(aoi_df.index, inplace=True)
    return df[principal_df.columns]


def get_loadings(input_df: DataFrame, pca: PCA):
    columns = get_pca_cols(pca.n_components_)
    components = pca.components_
    return DataFrame(components.T, columns=columns, index=input_df.columns)


def print_pca_stats(input_df: DataFrame, pca: PCA):
    print("Explained Variance Ratio %s" % pca.explained_variance_ratio_)
    print("\n")
    loadings_df = get_loadings(input_df, pca)
    display(loadings_df, title="Loadings")


def kmeans(df: DataFrame, n_clusters=3, random_state=0):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(df)
    df.loc[df.index, CLUSTER] = (kmeans.labels_.astype(int) + 1).astype(int)
    df[CLUSTER] = df[CLUSTER].astype(int)
    return df


def plot_scatter(df: DataFrame, x, y):
    clusters = Categorical(df[CLUSTER])
    colors = plt.cm.viridis(linspace(0, 1, len(clusters.categories)))  # Generate colors
    color_map = dict(zip(clusters.categories, colors))
    plt.figure(figsize=(10, 6))
    for cluster, color in color_map.items():
        subset = df[df[CLUSTER] == cluster]
        plt.scatter(subset[x], subset[y], c=[color], label=cluster, s=100)
    plt.legend(title="Cluster")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title('K-means Clustering')
    plt.show()


def map_columns_to_selected_aoi_masks(df: DataFrame) -> DataFrame:
    df[IS_LIE] = is_lie_selected(df, SELECTED_AOI)
    df[IS_TRUTH] = is_truth_selected(df, SELECTED_AOI)
    df[IS_OTHER] = is_other_selected(df, SELECTED_AOI)
    return df


def merge_components(aoi_df: DataFrame, pca_df: DataFrame):
    pca_df = pca_df.reset_index()
    aoi_df = aoi_df.reset_index()
    df = pd.merge(aoi_df[[SELECTED_AOI]], pca_df, left_index=True, right_index=True, how="left")
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


def get_kmeans_clusters(clustering_df: DataFrame, n_components: float = None, n_clusters: int = 4):
    if n_components:
        clustering_df = pca(clustering_df, n_components=n_components)
    kmeans_df = kmeans(clustering_df, n_clusters=n_clusters)
    return kmeans_df


def prepare_data(aoi_df: DataFrame, columns: List[str] = [SELF_LIE, OTHER_LIE, SELF_TRUE, OTHER_TRUTH], scaler=RobustScaler):
    filtered_df = aoi_df[columns] if columns else aoi_df[aoi_df.columns]
    normalised_df = scaler().fit_transform(filtered_df)
    return DataFrame(normalised_df, columns=filtered_df.columns)

def plot_correlation_matrix(for_kmeans_df: DataFrame, to_file: str = None):
    correlation_matrix = for_kmeans_df.corr()
    sns.heatmap(correlation_matrix, annot=True)
    if to_file:
        os.makedirs(os.path.dirname(to_file), exist_ok=True)
        plt.savefig(to_file)
    plt.show()
