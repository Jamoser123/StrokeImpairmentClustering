# funcs.py
import numpy as np
import pandas as pd
import torch
import itertools
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from src.models.contrastiveModels import TimeSeriesSimCLRWithAdversary, TimeSeriesSimCLR
from src.datasets.contrastiveDatasets import create_embeddingDataset, create_embeddingLoader
from src.util.funcs import set_all_seeds

from src.util.funcs import getMLFeatures
from src.clustering.util import plot_confusion_matrix, matchClusters, compareClusterings
from src.feature_calculation.preprocess import calc_alternative_ratios, order_clusters

# move visualize lowdim cluster somewhere else
def getEmbeddingCompdf(cfg): # Todo: add paths to cfg
    df_fma = pd.read_csv('/cluster/work/vogtlab/Group/jamoser/classifier_data/fma_clinical.csv')
    df_IMU = pd.read_csv('~/Notebooks/Data/data_merged_incl.GAIT_TH2_Recalculated_with_Jerk.csv')
    df_combined = pd.merge(df_fma, df_IMU, on=['Pat_id', 'visit'], how='left')
    
    df_combined = calc_alternative_ratios(df_combined)

    scaler = StandardScaler()
    scaled_fma = scaler.fit_transform(df_combined[['fma_ue_aff_total']])

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=cfg.ml.num_categories, random_state=cfg.seed)
    clusters_fma = kmeans.fit_predict(scaled_fma)
    df_combined['cluster_fma'] = clusters_fma

    df_combined = order_clusters(df_combined, 'cluster_fma', ['fma_ue_aff_total'])
    return df_combined

def calculateEmbeddings(cfg, model, loader):
    model.eval()

    # Dictionary to store embeddings
    embeddings_dict = {}
    
    with torch.no_grad():
        # Wrap loader with tqdm
        for batch in tqdm(loader, desc="Generating embeddings", unit="batch"):
            # Assuming batch is (timeseries, keys)
            timeseries, keys = batch[0], batch[1]
            
            # Get embeddings for the batch
            batch_embeddings = model(timeseries)
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

    # Add Pat_id and visit
    df['Pat_id'] = df['short_key'].apply(lambda x: x.split('/')[2][:5])
    df['visit'] = df['short_key'].apply(lambda x: x.split('/')[1])
    
    return df, embedding_matrix

def clusterEmbeddings(cfg, df, embedding_matrix):
    kmeans = KMeans(n_clusters=cfg.ml.num_categories, random_state=cfg.seed)
    cluster_labels = kmeans.fit_predict(embedding_matrix)
    df['cluster'] = cluster_labels
    return df

def projectEmbeddings(cfg, df, embedding_matrix):
    pca = PCA(n_components=cfg.ml.p_components)
    embeddings_kd = pca.fit_transform(embedding_matrix)
    column_names = [f'PC{i}' for i in range(1, cfg.ml.p_components + 1)]
    pca_df = pd.DataFrame(embeddings_kd, columns=column_names)

    # Concatenate the PCA DataFrame with the original DataFrame
    combined_df = pd.concat([df.reset_index(drop=True), pca_df], axis=1)
    return combined_df

def generatePCPairs(cfg):
    column_names = [f'PC{i}' for i in range(1, cfg.ml.p_components + 1)]
    pairs = list(itertools.combinations(column_names, 2))
    return pairs

def visualize_lowdimensional_clusters(cfg, 
                                    df: pd.DataFrame,
                                    x_col: str = 'PC1', 
                                    y_col: str = 'PC2',
                                    color_cols: list = ['cluster'],
                                    plot_days: list = None,
                                    titles: list = None) -> None:
    """
    Creates scatter plots of low-dimensional data, with one subplot for each color column.
    Can either display plots or save them to a specified directory.

    Args:
    - df (pd.DataFrame): DataFrame containing the data
    - cfg: Configuration object containing save settings
    - x_col (str): Column name for x-axis
    - y_col (str): Column name for y-axis
    - color_cols (list): List of column names to use for coloring points
    - plot_days (list): If specified, filters data to these days
    - titles (list): Optional list of titles for each subplot. If None, uses color_col names
    """
    # Filter by days if specified
    if plot_days:
        df = df[df['visit'].isin(plot_days)]
        print(f'\nPlot days {plot_days}')
    else:
        print(f'\nPlot all days')

    # Use provided titles or default to color_cols names
    if titles is None:
        titles = color_cols

    # Determine if saving plots
    save_plots = hasattr(cfg, 'save_directory') and cfg.save_directory is not None

    for color_col, title in zip(color_cols, titles):
        # Create new figure for each plot
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        unique_values = sorted(df[color_col].unique())
        for value in unique_values:
            mask = df[color_col] == value
            count = mask.sum()
            label = 'Low' if value == 0 else 'Medium' if value == 1 else 'High'
            plt.scatter(df.loc[mask, x_col], 
                       df.loc[mask, y_col], 
                       label=label,#f'{value}, n:{count}',
                       edgecolor='k',
                       s=50)

        # Set labels and title
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.legend(loc='upper right')
        plt.axis('equal')

        if save_plots:
            # Create filename from title
            filename = f"{title.lower().replace(' ', '_')}.png"
            save_path = os.path.join(cfg.save_directory, filename)
            # Ensure directory exists
            os.makedirs(cfg.save_directory, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def groupBy_PM_Pair(cfg, df):
    key_clusters = df.groupby('short_key').agg({
        'cluster': [
            ('final', lambda x: x.mode()[0]),
            ('count', 'count'),
            ('unique_values', lambda x: list(x.value_counts().items()))
        ],
        **{f'PC{i}': ('mean', 'mean') for i in range(1, cfg.ml.p_components + 1)}  # Adding averages for each principal component
    }).reset_index()

    # Flatten the MultiIndex columns and ensure unique names
    key_clusters.columns = [
        '_'.join(col).strip() if isinstance(col, tuple) else col for col in key_clusters.columns.values
    ]

    # Remove '_mean' suffix from principal component columns
    key_clusters.columns = [col.replace('_mean', '') for col in key_clusters.columns]

    # Remove duplicates by keeping only the first occurrence of each unique name
    key_clusters = key_clusters.loc[:, ~key_clusters.columns.duplicated()]

    key_clusters['Pat_id'] = key_clusters['short_key_'].apply(lambda x: x.split('/')[2][:5])
    key_clusters['visit'] = key_clusters['short_key_'].apply(lambda x: x.split('/')[1])
    
    return key_clusters

def merge_Embeddings_and_FMA(cfg, df, df_fma):

    # Now we can merge with the FMA dataframe
    df = df_fma.merge(df, on=['Pat_id', 'visit'], how='inner')

    df['cluster_final'] = df['cluster_final'].astype(int)
    df['cluster_fma'] = df['cluster_fma'].astype(int)

    return df

def embeddingPipeline(cfg):
    
    if cfg.ml.train_adversary:
        model = TimeSeriesSimCLRWithAdversary.load_from_checkpoint(cfg.checkpoint_path)
    else:
        model = TimeSeriesSimCLR.load_from_checkpoint(cfg.checkpoint_path)
    set_all_seeds(cfg.seed)

    feature_columns = getMLFeatures(cfg)
    
    dataset = create_embeddingDataset(cfg, feature_columns)
    loader = create_embeddingLoader(cfg, dataset)

    df, embedding_matrix = calculateEmbeddings(cfg, model, loader)
    
    # Add cluster labels to dataframe
    df = clusterEmbeddings(cfg, df, embedding_matrix)

    # Calculate k principle components
    df = projectEmbeddings(cfg, df, embedding_matrix)

    df_grouped = groupBy_PM_Pair(cfg, df)
    # PCA of averaged cluster assignment
    
    df_fma = getEmbeddingCompdf(cfg)
    df_merged = merge_Embeddings_and_FMA(cfg, df_grouped, df_fma)

    df_merged = matchClusters(cfg, df_merged, 'cluster_final', 'cluster_fma')

    compareClusterings(cfg, df_merged, 'cluster_final', 'cluster_fma')

    visualize_lowdimensional_clusters(cfg,
                                      df_merged,
                                      x_col='PC1',
                                      y_col='PC2',
                                      color_cols=['cluster_final', 'cluster_fma'],
                                      titles=['Representations (Deep Learning Clustering)', 'Representations (Clinical Clustering)']#First two PC-Embeddings colored by Kmeans on Embeddings','First two PC-Embeddings colored by FMA category']
                                     )
    return df, df_grouped, df_merged