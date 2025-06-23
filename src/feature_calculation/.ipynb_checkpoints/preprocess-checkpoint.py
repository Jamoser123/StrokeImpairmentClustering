# preprocess.py
import pandas as pd
import numpy as np

def order_clusters(df, cluster_col, feature_cols):
    cluster_means = df.groupby(cluster_col)[feature_cols].mean()
    
    if len(feature_cols) > 1:
        cluster_values = cluster_means.apply(np.linalg.norm, axis=1)
    else:
        cluster_values = cluster_means[feature_cols[0]]
    
    ordered_clusters = cluster_values.sort_values().index
    
    cluster_mapping = {old: new for new, old in enumerate(ordered_clusters)}
    
    df[cluster_col] = df[cluster_col].map(cluster_mapping)
    return df

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