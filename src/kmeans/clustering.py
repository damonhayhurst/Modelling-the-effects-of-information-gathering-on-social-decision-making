from numpy import linspace
from pandas import Categorical, DataFrame, merge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils.display import display
from utils.read_csv import read_from_analysis_file
from utils.paths import AOI_ANALYSIS_CSV
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils.columns import *

DWELL_DICT = {column: index for index, column in enumerate(DWELL_COLUMNS)}


def get_pca_col(i): return "Principal Component %s" % i
def get_pca_cols(n): return [get_pca_col(i) for i in range(1, n + 1)]


def create_for_clustering(df: DataFrame, columns: list[str] = []):
    df = df.reset_index()
    numerical_df = map_columns_to_numerical(df)
    filtered_df = numerical_df[columns]
    return filtered_df


def map_columns_to_numerical(df: DataFrame):
    df[FIRST_AOI] = df[FIRST_AOI].map(DWELL_DICT)
    df[LAST_AOI] = df[LAST_AOI].map(DWELL_DICT)
    return df


def pca(df: DataFrame, n_components=0.9) -> DataFrame:
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    columns = get_pca_cols(pca.n_components_)
    principal_df = DataFrame(data=principal_components, columns=columns)
    print_pca_stats(df, pca)
    return principal_df


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
    df[CLUSTER] = kmeans.labels_
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


def do_clustering(df: DataFrame,  n_components: int | float = 0, n_clusters: int = 4, columns: list[str] = [*DWELL_COLUMNS, PAYNE_INDEX]):
    clustering_df = create_for_clustering(df, columns=columns)
    if n_components != 0:
        clustering_df = pca(clustering_df, n_components=n_components)
    kmeans_df = kmeans(clustering_df, n_clusters=n_clusters)
    return kmeans_df


if __name__ == '__main__':
    df = read_from_analysis_file(AOI_ANALYSIS_CSV)
    cluster_df = do_clustering(df, n_components=0.9).pipe(display)

    plot_scatter(cluster_df, x=get_pca_col(1), y=get_pca_col(2))
