import pandas as pd
import numpy as np
from tslearn.metrics import cdist_dtw, cdist_soft_dtw, cdist_gak
from sklearn.cluster import HDBSCAN
import matplotlib.pyplot as plt

def LoadData(filepath, sep='\t'):
    df = pd.read_csv(filepath, sep=sep)
    return df

def GetData(df, concentration_cols, normalize=True, smooth=True, window_size=100):
    """
    Extract, normalize, and optionally smooth data from the DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - concentration_cols: List of column names to extract.
    - normalize: Boolean to turn normalizatsklaerion on or off.
    - smooth: Boolean to turn smoothing on or off.
    - window_size: Size of the rolling window for smoothing.

    Returns:
    - data_array: 2D numpy array of the data.
    """
    data_gathered = []

    for col in concentration_cols:
        data = df[col].values

        # Normalize the data
        if normalize:
            data = (data - data.mean()) / data.std()

        # Smooth the data
        if smooth:
            data_smoothed = pd.Series(data).rolling(window=window_size, center=True).mean().dropna().values
            data_gathered.append(data_smoothed)
        else:
            data_gathered.append(data)

    # Ensure all series have the same length
    min_length = min(len(series) for series in data_gathered)
    data_gathered = [series[:min_length] for series in data_gathered]

    # Stack the data vertically
    data_array = np.vstack(data_gathered)

    return data_array


def ComputeTSDistance(data, metrics=['dtw', 'soft_dtw', 'p4']):
    """
    Compute pairwise distances between time series using various metrics.

    Parameters:
    - data: 2D numpy array of time series data.
    - metrics: List of distance metrics to compute.

    Returns:
    - distances: Dictionary of distance matrices for each metric.
    """
    distances = {}

    if 'dtw' in metrics:
        distances['dtw'] = cdist_dtw(data)

    if 'soft_dtw' in metrics:
        mat = cdist_soft_dtw(data)
        distances['soft_dtw'] = mat - np.min(mat)

    if 'p4' in metrics:
        rows = data.shape[0]
        res = np.zeros(shape=(rows, rows), dtype=np.float64)
        for i in range(rows):
            row_i = data[i,:]
            for j in range(data.shape[0]):
                row_j = data[j,:]
                if i == j:
                    continue
                elif i < j:
                    continue
                else:
                    order = 4
                    norm = np.pow(np.abs(row_i - row_j), order)
                    res[i, j] = np.pow(np.sum(norm), 1/order)
                    res[j, i] = res[i, j]
        distances['p4'] = res
                
    return distances


def PerformHDBSCAN(distance_matrix, min_samples=2):

    # Perform DBSCAN
    hdbscan = HDBSCAN(min_cluster_size=2, min_samples=min_samples, metric='precomputed')
    labels = hdbscan.fit_predict(distance_matrix)

    # Print and return labels
    print("HDBSCAN Labels:", labels)
    return labels


def PlotClusterRows(data, concentration_cols, labels, title, filename):
    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))

    # Create a figure with subplots for each cluster
    fig, axes = plt.subplots(n_clusters, 1, figsize=(12, 3 * n_clusters), sharex=True)

    if n_clusters == 1:
        axes = [axes]  # Ensure axes is iterable even if there's only one subplot

    for ax, label in zip(axes, unique_labels):
        for i, col in enumerate(concentration_cols):
            if labels[i] == label:
                ax.plot(data[i, :], label=col, color=colors[list(unique_labels).index(label)], linewidth=2)
        ax.set_title(f'Cluster {label}')
        ax.set_ylabel('Concentration')
        ax.grid(True)
        ax.legend()

    axes[-1].set_xlabel('Time (aligned)')
    fig.suptitle(title, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout to reduce whitespace
    plt.savefig(filename, format='jpg', bbox_inches='tight')
    plt.close()


def main():
    # Load data
    df = LoadData('260113_Vanillin+UV.txt')

    # Identify concentration columns
    concentration_cols = [col for col in df.columns if col.startswith('m') and '(' in col] # The name of the time series
    data_array = GetData(df, concentration_cols, smooth=False, window_size=100)
    smooth_data_array = GetData(df, concentration_cols, smooth=True, window_size=100)

    # Compute Distance measures
    distance_matrices = ComputeTSDistance(data_array)
    smooth_distance_matrices = ComputeTSDistance(smooth_data_array)

    # Do clustering and plot the result
    for label, d_mat in distance_matrices.items():
        hdbscan_labels= PerformHDBSCAN(d_mat) # Element x in concentration_cols belongs to cluster i where i is element x in hdbscan_labels
        PlotClusterRows(data_array, concentration_cols, hdbscan_labels, f'HDBSCAN Clustering: {label}', f'hdbscan_clusters_{label}_raw.pdf')

    for label, d_mat in smooth_distance_matrices.items():
        hdbscan_labels= PerformHDBSCAN(d_mat)
        PlotClusterRows(smooth_data_array, concentration_cols, hdbscan_labels, f'HDBSCAN Clustering: {label}', f'hdbscan_clusters_{label}_smooth.pdf')


if __name__ == "__main__":
    main()
