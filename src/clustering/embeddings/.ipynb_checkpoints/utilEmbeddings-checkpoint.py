import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.optimize import linear_sum_assignment
from scipy.stats import multivariate_normal, spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from typing import Union, List, Optional
import torch
from tqdm import tqdm
from util.model import TimeSeriesSimCLR
from util.dataset import create_embeddingDataset, create_embeddingLoader
from util.utilFuncs import set_all_seeds, split_patients
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import os

def calc_alternative_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate alternative ratio metrics
    
    Args:
        df (pd.DataFrame): The dataset containing features to calculate new ratios
        
    Returns:
        pd.DataFrame: A DataFrame with the new ratios
    """

    df['use_Dur_ratio_alt'] = (df['use_Dur_aff'] - df['use_Dur_nonaff'])/(df['use_Dur_aff'] + df['use_Dur_nonaff'])
    df['use_Dur_unilat_ratio_alt'] = (df['use_Dur_unilat_aff'] - df['use_Dur_unilat_nonaff'])/(df['use_Dur_unilat_aff'] + df['use_Dur_unilat_nonaff'])
    df['mean_Magn_ratio_alt'] = (df['mean_AC_aff'] - df['mean_AC_nonaff'])/(df['mean_AC_aff'] + df['mean_AC_nonaff'])
    df['Median_Magn_ratio_alt'] = (df['Median_AC_aff'] - df['Median_AC_nonaff'])/(df['Median_AC_aff'] + df['Median_AC_nonaff'])
    df['unilat_Magn_ratio_alt'] = (df['mean_AC_unilat_aff'] - df['mean_AC_unilat_nonaff'])/(df['mean_AC_unilat_aff'] + df['mean_AC_unilat_nonaff'])

    df['mean_Jerk_ratio'] = df['mean_Jerk_aff']/df['mean_Jerk_nonaff']
    df['median_Jerk_ratio'] = df['median_Jerk_aff']/df['median_Jerk_nonaff']
    df['mean_Jerk_ratio_alt'] = (df['mean_Jerk_aff'] - df['mean_Jerk_nonaff'])/(df['mean_Jerk_aff'] + df['mean_Jerk_nonaff'])
    df['median_Jerk_ratio_alt'] = (df['median_Jerk_aff'] - df['median_Jerk_nonaff'])/(df['median_Jerk_aff'] + df['median_Jerk_nonaff'])

    return df
# Step 1: Analyze all cluster pairings
def analyze_cluster_pairings(df, arat_fma_cluster_col, imu_cluster_col, imu_features):
    """
    Analyze cluster pairings between ARAT/FMA clusters and IMU-based clusters, compare IMU-derived features,
    and calculate the disagreement score (absolute difference between ARAT/FMA cluster and IMU-based cluster).
    
    Args:
        df (pd.DataFrame): DataFrame containing cluster assignments and IMU-derived features.
        arat_fma_cluster_col (str): Column name for ARAT/FMA cluster assignments.
        imu_cluster_col (str): Column name for IMU-based cluster assignments.
        imu_features (list): List of IMU-derived feature columns to compare.
    
    Returns:
        pd.DataFrame: DataFrame summarizing the mean and variance of IMU features for each cluster pairing,
                      along with the disagreement score and count of entries per group.
    """
    # Group by the cluster pairings
    cluster_group = df.groupby([arat_fma_cluster_col, imu_cluster_col])
    
    # Calculate the mean and variance of each IMU feature
    mean_df = cluster_group[imu_features].mean().reset_index()
    var_df = cluster_group[imu_features].var().reset_index()
    
    # Add a column for the count of entries in each group
    mean_df['count'] = cluster_group.size().reset_index(name='count')['count']

    # Print the disagreement scores and counts
    print(f"\nCounts:\n{mean_df[[arat_fma_cluster_col, imu_cluster_col, 'count']]}")
    
    # Print mean, variance, disagreement score, and count for each feature
    for feature in imu_features:
        print(f"\nFeature: {feature}")
        print(f"Mean values:\n{mean_df[[arat_fma_cluster_col, imu_cluster_col, feature]]}")
        print(f"Variance values:\n{var_df[[arat_fma_cluster_col, imu_cluster_col, feature]]}")
    
    # Return both the mean and variance DataFrames for further analysis
    return mean_df, var_df


# Step 2: Visualize the IMU-derived features for each cluster pairing
def visualize_cluster_pairings(df, arat_fma_cluster_col, imu_cluster_col, imu_features):
    """
    Visualize the IMU-derived features for each ARAT/FMA-IMU cluster pairing using boxplots.
    
    Args:
        df (pd.DataFrame): DataFrame containing cluster assignments and IMU-derived features.
        imu_features (list): List of IMU-derived feature columns to visualize.
        arat_fma_cluster_col (str): Column name for ARAT/FMA cluster assignments.
        imu_cluster_col (str): Column name for IMU-based cluster assignments.
    """
    for feature in imu_features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x=arat_fma_cluster_col, y=feature, hue=imu_cluster_col)
        plt.title(f"Distribution of {feature} by ARAT/FMA and IMU Clusters")
    plt.show()
    
def print_cluster_stats(df: pd.DataFrame, cluster_col: str, feature_names: List[str]) -> None:
    """
    Prints the mean and range (min, max) of all features for each cluster.

    Args:
    - df (pd.DataFrame): Input DataFrame containing features and a clustering column.
    - cluster_col (str): Column name that specifies the cluster labels.

    Returns:
    - None
    """
    # Group the DataFrame by the specified cluster column
    df = df[feature_names + [cluster_col]]
    grouped = df.groupby(cluster_col)
    
    for cluster, group in grouped:
        print(f"\nCluster {cluster}:")
        print("-" * 20)
        
        # Calculate and print the mean for each feature in the cluster
        print("Mean values:")
        mean_values = group.mean()
        print(mean_values)
        
        # Calculate and print the range (min, max) for each feature in the cluster
        print("\nRanges (min, max):")
        range_values = group.agg(['min', 'max'])
        print(range_values)
        
        print("\n" + "=" * 40)


def plot_pairplot_with_clusters(df: pd.DataFrame, 
                                     cluster_col: str,
                                     plot_days: list = ['day90'],
                                     cols_to_plot: Optional[list] = None,
                                     show_correlation: bool = True) -> None:
    """
    Visualizes a pairplot of the specified columns colored by clusters, with options to show
    correlation coefficients and remove the upper triangle.

    Args:
    - df (pd.DataFrame): DataFrame containing the data and cluster labels.
    - cluster_col (str): Column name for the cluster labels.
    - cols_to_plot (list, optional): List of column names to include in the pairplot. If None, all numeric columns are used.
    - show_correlation (bool): If True, displays the Spearman correlation coefficients on the upper triangle.
    
    Returns:
    None
    """
    if plot_days != []:
        df = df[df['visit'].isin(plot_days)]
        
    # If no specific columns are provided, use all numeric columns in the DataFrame
    if cols_to_plot is None:
        cols_to_plot = df.select_dtypes(include=np.number).columns.tolist()
    
    # Create the pairplot
    pairplot = sns.pairplot(df, hue=cluster_col, vars=cols_to_plot, diag_kind='kde')#, corner=True)

    # Optionally add correlation coefficients on the upper triangle
    if show_correlation:
        for i, j in zip(*np.triu_indices_from(pairplot.axes, 1)):
            x_col = cols_to_plot[j]
            y_col = cols_to_plot[i]
            corr, _ = spearmanr(df[x_col], df[y_col])
            pairplot.axes[i, j].annotate(f'Corr: {corr:.3f}', 
                                         xy=(0.5, 0.9), 
                                         xycoords='axes fraction',
                                         ha='center', 
                                         va='center', 
                                         fontsize=12, 
                                         color='black')
            pairplot.axes[i, j].set_axis_off()  # Turn off the upper triangle axes

    else:
        for i, j in zip(*np.triu_indices_from(pairplot.axes, 1)):
            pairplot.axes[i, j].set_visible(False)
    
    plt.tight_layout()

    plt.show()


def plot_confusion_matrix(crosstab, save_path=None):
    """
    Plot the confusion matrix calculated using pd.crosstab.
    If save_path is provided, save the plot as an image, otherwise plot to the output.
    
    Args:
        crosstab (pd.DataFrame): The confusion matrix calculated using pd.crosstab.
        save_path (str or None): Path to save the image. If None, the plot is displayed.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(crosstab, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel(crosstab.index.name or 'Actual')
    plt.xlabel(crosstab.columns.name or 'Predicted')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
        
def compare_clusters(data, valid_cluster_col, novel_cluster_col, print_worst=False, save_path=None, feature_list=[]):
    """
    Compares two sets of cluster labels using ARI, NMI, and accuracy scores, reassigns the cluster labels based on the best matching, 
    and returns the updated dataset with reordered cluster labels.

    Args:
    - data (pd.DataFrame): The dataset containing the cluster labels to be compared.
    - valid_cluster_col (str): Column name for the valid cluster labels.
    - novel_cluster_col (str): Column name for the novel cluster labels.
    - print_worst (bool): print information about patients that have heavy mismatch
    - save_path (str, optional): Path to save the confusion matrix plot. If None, the plot is displayed.

    Returns:
    - pd.DataFrame: The dataset with the reassigned cluster labels in the 'novel_cluster_col'.
    """
    
    # Extract the true and predicted labels
    y_true = data[valid_cluster_col].to_numpy()
    y_pred = data[novel_cluster_col].to_numpy()
    
    # Compute ARI and NMI
    ari_score = adjusted_rand_score(y_true, y_pred)
    nmi_score = normalized_mutual_info_score(y_true, y_pred)
    
    # Create the cost matrix for the Hungarian algorithm
    cost_matrix = np.zeros((len(set(y_true)), len(set(y_pred))))
    
    for i, valid_cluster in enumerate(set(y_true)):
        for j, novel_cluster in enumerate(set(y_pred)):
            cost_matrix[i, j] = -np.sum((y_true == valid_cluster) & (y_pred == novel_cluster))
    
    # Use the Hungarian algorithm to find the best cluster matching
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_clusters = list(zip(row_ind, col_ind))
    
    # Reorder the predicted labels according to the best matching
    y_pred_reordered = np.zeros_like(y_pred)
    for i, j in zip(row_ind, col_ind):
        y_pred_reordered[y_pred == j] = i
    
    # Update the dataset with the reassigned cluster labels
    data[novel_cluster_col] = y_pred_reordered
    
    # Calculate the confusion matrix with the reordered predictions
    confusion = pd.crosstab(y_true, y_pred_reordered, rownames=[valid_cluster_col], colnames=[novel_cluster_col])
    
    # Calculate Cluster Accuracy
    correct_assignments = np.sum(np.diag(confusion.values))
    total_assignments = np.sum(confusion.values)
    cluster_accuracy = correct_assignments / total_assignments

    if print_worst:
        
        best_cluster = 0  # Modify as per the definition of best cluster
        worst_cluster = max(set(y_true))  # Modify as per the definition of worst cluster, assuming max is worst
        
        # Find patients who have the best in one cluster and worst in the other
        mismatched_patients_weWorst = data[(data[valid_cluster_col] == best_cluster) & (data[novel_cluster_col] == worst_cluster)]
        mismatched_patients_weBest = data[(data[valid_cluster_col] == worst_cluster) & (data[novel_cluster_col] == best_cluster)]
    
        # Print the patient IDs of these mismatched patients
        if not mismatched_patients_weWorst.empty:
            print("Patients with the worst category in Arat/FMA and the best in own Clustering:")
            for idx, row in mismatched_patients_weWorst.iterrows():
                print(f"Patient ID: {row['Pat_id']}, Day: {row['visit']}")
                for feature in feature_list:
                    print(f"  {feature}: {row[feature]}")
                print("-" * 30) 
        else:
            print("No patients found with the worst in Arat/FMA and the best in own Clustering.")
    
        if not mismatched_patients_weBest.empty:
            print("Patients with the best category in Arat/FMA and the worst in own Clustering:")
            for idx, row in mismatched_patients_weBest.iterrows():
                print(f"Patient ID: {row['Pat_id']}, Day: {row['visit']}")
                for feature in feature_list:
                    print(f"  {feature}: {row[feature]}")
                print("-" * 30) 
        else:
            print("No patients found with the best in Arat/FMA and the worst in own Clustering.")
    
    # Print the results
    print(f"Adjusted Rand Index (ARI): {ari_score}")
    print(f"Normalized Mutual Information (NMI): {nmi_score}")
    print(f"Cluster Accuracy: {cluster_accuracy:.4f}")
    print(f"Best Cluster Matching: {matched_clusters}")
    print("Confusion Matrix:")
    plot_confusion_matrix(confusion, save_path)
    
    return data

def visualize_lowdimensional_clusters(df: pd.DataFrame, 
                                      model: Union[KMeans, GaussianMixture],
                                      plot_days: list = ['day90'],
                                      x_col: str = 'Component_1', 
                                      y_col: str = 'Component_2', 
                                      cluster_col: str = 'cluster_kmeans', 
                                      second_cluster: str = None,
                                      save_path: str = None) -> None:
    """
    Visualizes low-dimensional data colored by clusters, and plots centroids for KMeans 
    or Gaussian distributions for GMM. Additionally, a second plot shows the data colored 
    by another clustering if provided.

    Args:
    - df (pd.DataFrame): DataFrame containing the low-dimensional components and cluster labels.
    - model: A fitted clustering model (KMeans or GaussianMixture).
    - x_col (str): Column name for the x-axis (Component 1).
    - y_col (str): Column name for the y-axis (Component 2).
    - cluster_col (str): Column name for the cluster labels.
    - second_cluster (str): Column name for a second set of cluster labels. If provided, a second scatter plot is created.
    - save_path (str): If specified, saves the plot to this path. Otherwise, displays the plot.
    
    Returns:
    None
    """
    if plot_days != []:
        df = df[df['visit'].isin(plot_days)]
        print(f'\nPlot days {plot_days}')
    else:
        print(f'\nPlot all days')
        
    total_count = len(df)
    
    if second_cluster:
        gs = gridspec.GridSpec(1, 2)
        plt.figure(figsize=(14, 6))
    else:
        plt.figure(figsize=(7, 6))
        gs = gridspec.GridSpec(1, 1)
    
    # First plot: points colored by primary cluster
    ax0 = plt.subplot(gs[0])
    unique_clusters = sorted(df[cluster_col].unique())
    for cluster in unique_clusters:
        cluster_data = df[df[cluster_col] == cluster]
        count = len(cluster_data)
        ax0.scatter(cluster_data[x_col], cluster_data[y_col], label=f'Cluster {cluster}, n:{count}', edgecolor='k', s=50)
    
    if isinstance(model, KMeans) and model.cluster_centers_.shape[1] == 2:
        # Plot centroids for KMeans
        centroids = model.cluster_centers_
        ax0.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')
        
    elif isinstance(model, GaussianMixture) and model.means_.shape[1] == 2:
        # Plot Gaussian ellipses for GMM
        means = model.means_
        covariances = model.covariances_
        for i, (mean, cov) in enumerate(zip(means, covariances)):
            # Add contour lines for each Gaussian component
            draw_contour(mean, cov, ax=ax0)


    ax0.set_xlabel(x_col)
    ax0.set_ylabel(y_col)
    ax0.axis('equal')  # Ensure even scaling
    ax0.set_title(f'Clusters on Reduced Data ({cluster_col})')
    ax0.legend(loc='upper right')

    # Second plot: points colored by the second cluster
    if second_cluster:
        ax1 = plt.subplot(gs[1])
        unique_second_clusters = sorted(df[second_cluster].unique())
        for cluster in unique_second_clusters:
            cluster_data = df[df[second_cluster] == cluster]
            count = len(cluster_data)
            ax1.scatter(cluster_data[x_col], cluster_data[y_col], label=f'{second_cluster} {cluster}, n:{count}', edgecolor='k', s=50)
        
        ax1.set_xlabel(x_col)
        ax1.set_ylabel(y_col)
        ax1.axis('equal')  # Ensure even scaling
        ax1.set_title(f'Clusters from {second_cluster} on Reduced Data')
        ax1.legend(loc='upper right')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def draw_contour(mean: np.ndarray, cov: np.ndarray, ax: plt.Axes) -> None:
    """
    Draws contour lines representing a Gaussian distribution.
    
    Args:
    - mean (np.ndarray): Mean of the Gaussian distribution.
    - cov (np.ndarray): Covariance matrix of the Gaussian distribution.
    - ax (plt.Axes): Matplotlib Axes object to draw the contour lines on.
    
    Returns:
    None
    """    
    x, y = np.mgrid[mean[0]-3*np.sqrt(cov[0,0]):mean[0]+3*np.sqrt(cov[0,0]):.01, 
                    mean[1]-3*np.sqrt(cov[1,1]):mean[1]+3*np.sqrt(cov[1,1]):.01]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    levels = np.linspace(0, rv.pdf(mean), 6)
    ax.contour(x, y, rv.pdf(pos), levels=levels, cmap='Reds', alpha=0.8)

def embeddingPipeline(cfg, features_to_plot):
    model = TimeSeriesSimCLR.load_from_checkpoint(cfg.checkpoint_path)
    set_all_seeds(cfg.seed)

    feature_columns = [
    'wrist_aff__acc_x', 'wrist_aff__acc_y', 'wrist_aff__acc_z',
    'wrist_aff__gyro_x', 'wrist_aff__gyro_y', 'wrist_aff__gyro_z',
    'wrist_nonaff__acc_x', 'wrist_nonaff__acc_y', 'wrist_nonaff__acc_z',
    'wrist_nonaff__gyro_x', 'wrist_nonaff__gyro_y', 'wrist_nonaff__gyro_z'
    ] if cfg.ml.useGyro else [
    'wrist_aff__acc_x', 'wrist_aff__acc_y', 'wrist_aff__acc_z',
    'wrist_nonaff__acc_x', 'wrist_nonaff__acc_y', 'wrist_nonaff__acc_z',
    ]
    
    dataset = create_embeddingDataset(cfg, feature_columns)
    loader = create_embeddingLoader(cfg, dataset)

    model.eval()

    # Dictionary to store embeddings
    embeddings_dict = {}
    
    with torch.no_grad():
        # Wrap loader with tqdm
        for batch in tqdm(loader, desc="Generating embeddings", unit="batch"):
            # Assuming batch is (timeseries, keys)
            timeseries, keys = batch[0], batch[1]
            
            # Get embeddings for the batch
            batch_embeddings = model.get_embedding(timeseries)
            # Move embeddings to CPU and convert to numpy for storage
            batch_embeddings = batch_embeddings.cpu().numpy()
            
            # Store embeddings by key
            for key, embedding in zip(keys, batch_embeddings):
                if key not in embeddings_dict:
                    embeddings_dict[key] = []
                embeddings_dict[key].append(embedding)
    
    rows = []
    for key, embeddings in embeddings_dict.items():
        for embedding in embeddings:
            # Create the original key and short key (without T1 suffix)
            short_key = '/'.join(key.split('_')[0].split('/'))  # This removes everything after '_'
            rows.append({
                'key': key,
                'short_key': short_key,
                'embedding': embedding
            })
    
    # Create dataframe
    df = pd.DataFrame(rows)

    embedding_matrix = np.stack(df['embedding'].apply(lambda x: np.array(x)).values)

    kmeans = KMeans(n_clusters=3, random_state=42)
    cluster_labels = kmeans.fit_predict(embedding_matrix)
    
    # Add cluster labels to dataframe
    df['cluster'] = cluster_labels

    # Perform PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embedding_matrix)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=df['cluster'], cmap='viridis', alpha=0.6)
    
    # Add colorbar
    plt.colorbar(scatter, label='Cluster')
    
    # Add labels and title
    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title('Clustering Results on all time slices (PCA visualization)')
    
    # Add total explained variance
    total_var = pca.explained_variance_ratio_.sum()
    plt.suptitle(f'Total explained variance: {total_var:.3f}', y=1.02)
    
    plt.tight_layout()

    # Set fixed axis range
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    plt.axis('equal')
    plt.show()

    key_clusters = df.groupby('short_key')['cluster'].agg(lambda x: x.mode()[0]).reset_index()
    key_clusters.columns = ['short_key', 'final_cluster']

    # Get the mode (most common) cluster and count for each key
    key_clusters = df.groupby('short_key').agg({
        'cluster': [
            ('final_cluster', lambda x: x.mode()[0]),
            ('cluster_count', 'count'),
            ('unique_values', lambda x: list(x.value_counts().items()))  # This gives us [(cluster_value, count), ...]
        ]
    }).reset_index()
    
    # Flatten column names
    key_clusters.columns = ['short_key', 'final_cluster', 'cluster_count', 'cluster_distribution']

    # PCA of averaged cluster assignment
    pca_df = pd.DataFrame(embeddings_2d, columns=['PC1', 'PC2'])
    pca_df['short_key'] = df['short_key'].values
    
    # Calculate mean position for each key
    key_positions = pca_df.groupby('short_key')[['PC1', 'PC2']].mean().reset_index()
    
    # Merge with the final cluster assignments
    key_positions = key_positions.merge(key_clusters, on='short_key', how='left')
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(key_positions['PC1'], key_positions['PC2'], 
                         c=key_positions['final_cluster'], cmap='viridis', 
                         s=100, alpha=0.8)
    
    # Add colorbar
    plt.colorbar(scatter, label='Cluster')
    
    # Add labels and title
    plt.xlabel(f'First Principal Component (explained variance: {pca.explained_variance_ratio_[0]:.3f})')
    plt.ylabel(f'Second Principal Component (explained variance: {pca.explained_variance_ratio_[1]:.3f})')
    plt.title('Final Clustering Results (PCA visualization)')
    
    # Add total explained variance
    total_var = pca.explained_variance_ratio_.sum()
    plt.suptitle(f'Total explained variance: {total_var:.3f}', y=1.02)
    
    plt.tight_layout()
    
    # Set fixed axis range
    # plt.xlim(-2, 2)
    # plt.ylim(-2, 2)
    plt.axis('equal')
    plt.show()

    fma_df = pd.read_csv('/cluster/work/vogtlab/Group/jamoser/classifier_data/fma_clinical.csv')
    fma_df = fma_df[['Pat_id', 'visit', 'fma_ue_aff_total']]
    
    category_ranges = {
        0: (0, 23),    
        1: (23, 46),   
        2: (46, 67)   
    }
    
    # Drop rows with missing FMA values
    fma_df = fma_df.dropna(subset=['fma_ue_aff_total'])
    
    # Function to assign category based on the ranges
    def assign_category(value):
        for category, (lower, upper) in category_ranges.items():
            if lower <= value < upper:
                return category
        return None  # for any values outside all ranges
    
    # Add new column with categories
    fma_df['fma_category'] = fma_df['fma_ue_aff_total'].apply(assign_category)

    df_IMU = pd.read_csv('~/Notebooks/Data/data_merged_incl.GAIT_TH2_Recalculated_with_Jerk.csv')

    fma_df = pd.merge(df_IMU, fma_df, on=['Pat_id', 'visit'])

    fma_df = calc_alternative_ratios(fma_df)
    
    key_positions['Pat_id'] = key_positions['short_key'].apply(lambda x: x.split('/')[2][:5])  # Extract RU016
    key_positions['visit'] = key_positions['short_key'].apply(lambda x: x.split('/')[1])       # Extract day28
    
    # Now we can merge with the FMA dataframe
    comparison_df = fma_df.merge(key_positions[['Pat_id', 'visit', 'final_cluster', 'PC1', 'PC2', 'cluster_distribution']], 
                                on=['Pat_id', 'visit'], 
                                how='inner')

    comparison_df['final_cluster'] = comparison_df['final_cluster'].astype(int)
    comparison_df['fma_category'] = comparison_df['fma_category'].astype(int)
    
    # Call the comparison function
    comparison_df = compare_clusters(
        data=comparison_df,
        valid_cluster_col='fma_category',
        novel_cluster_col='final_cluster',
        # print_worst=True,
        # feature_list=['cluster_distribution']  # Add any other features you want to see for mismatched cases
    )

    visualize_lowdimensional_clusters(comparison_df, 
                                  model=model,
                                  plot_days=[],
                                  x_col='PC1',
                                  y_col='PC2',
                                  cluster_col='final_cluster',
                                  second_cluster='fma_category')
    
    return comparison_df
        
    #print_cluster_stats(df=comparison_df, cluster_col='final_cluster', feature_names=features_to_plot)
    #plot_pairplot_with_clusters(df=comparison_df, cluster_col='final_cluster', plot_days=[], cols_to_plot=features_to_plot, show_correlation=False)

    #analyze_cluster_pairings(comparison_df, 'final_cluster', 'fma_category', features_to_plot)
    #visualize_cluster_pairings(comparison_df, 'final_cluster', 'fma_category', features_to_plot)