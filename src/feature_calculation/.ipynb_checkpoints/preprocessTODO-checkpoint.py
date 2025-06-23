def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def explore_dataset(df: pd.DataFrame):
    """
    Explore the dataset by displaying its basic information and statistics.
    
    Args:
        df (pd.DataFrame): The dataset to explore.
    """
    print("First 5 rows of the dataset:")
    print(df.head(), "\n")
    
    print("Dataset info:")
    print(df.info(), "\n")
    
    print("Summary statistics of the dataset:")
    print(df.describe(), "\n")
    
    print("Checking for missing values:")
    print(df.isnull().sum(), "\n")

def categorize_arat_fma(data):
    """
    This function categorizes the ARAT and FMA-UE scores based on the provided ranges.
    It adds two new columns to the dataset: 'arat_category' and 'fma_category'.
    
    Parameters:
    data (DataFrame): The pandas DataFrame containing the columns 'arat_aff_total' and 'fma_ue_aff_total'.
    
    Returns:
    DataFrame: The original DataFrame with the added 'arat_category' and 'fma_category' columns.
    """
    
    # Define the categories for ARAT based on FMA-UE ranges
    def get_fma_category(fma_score):
        if pd.isna(fma_score):
            return ''
        elif 0 <= fma_score <= 22:
            return 0#'0.No'# upper-limb capacity'
        elif 23 <= fma_score <= 31:
            return 1#'1.Poor'# capacity'
        elif 32 <= fma_score <= 47:
            return 2#'2.Limited'# capacity'
        elif 48 <= fma_score <= 52:
            return 3#'3.Notable'# capacity'
        elif 53 <= fma_score <= 66:
            return 4#'4.Full'# upper-limb capacity'
        else:
            return ''
    
    def get_arat_category(arat_score):
        if pd.isna(arat_score):
            return ''
        elif 0 <= arat_score <= 10:
            return 0#'0.No'# upper-limb capacity'
        elif 11 <= arat_score <= 21:
            return 1#'1.Poor'# capacity'
        elif 22 <= arat_score <= 42:
            return 2#'2.Limited'# capacity'
        elif 43 <= arat_score <= 54:
            return 3#'3.Notable'# capacity'
        elif 55 <= arat_score <= 57:
            return 4#'4.Full'# upper-limb capacity'
        else:
            return ''

    # Apply the categorization functions to the columns if they exist
    if 'fma_ue_aff_total' in data.columns:
        data['fma_category'] = data['fma_ue_aff_total'].apply(get_fma_category)
    
    if 'arat_aff_total' in data.columns:
        data['arat_category'] = data['arat_aff_total'].apply(get_arat_category)

    
    return data

# Clustering
def cluster_data(data, feature_columns, k=3, day=None):
    """
    This function performs KMeans clustering on the specified feature columns in the dataset.
    
    Parameters:
    data (DataFrame): The pandas DataFrame containing the feature columns to be used for clustering.
    feature_columns (list): A list of column names to be used for clustering.
    k (int, optional): The number of clusters to use for KMeans. Default is 3.
    day (int, optional): If specified, filters the data to only include rows with 'visit_d' equal to this value. 
                         If not set, all data is used.
   
    
    Returns:
    KMeans: The KMeans model trained on the selected features.
    """
    if day is not None:
        data = data[data['visit_d'] == day]
    
    # Extract the specified feature columns for clustering
    features = data[feature_columns]

    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)

     # Get the cluster centers and sort them
    #centers = sorted(kmeans.cluster_centers_.flatten())
    
    # Calculate the midpoints between consecutive centers
    #boundaries = [(centers[i] + centers[i+1])/2 for i in range(len(centers)-1)]
    #print(boundaries)
    
    return kmeans

def predict_clusters(data, cluster_col, feature_columns, kmeans_model):
    """
    This function predicts the clusters using a pre-fitted KMeans model based on the specified feature columns.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the feature columns to be used for prediction.
    kmeans_model (KMeans): A pre-fitted KMeans model.
    feature_columns (list): A list of column names to be used for prediction.

    Returns:
    DataFrame: A copy of the original DataFrame with an additional column 'cluster_pca' indicating the predicted cluster for each row.
    """
    # Extract the specified feature columns for prediction
    features = data[feature_columns]

    # Predict the clusters using the KMeans model
    cluster_labels = kmeans_model.predict(features)

    # Assign the cluster labels to a new column
    data[cluster_col] = cluster_labels

    return data

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

# Log transformation with bounds
def log_transform_features(df: pd.DataFrame, feature_columns: list, lower_bound: float = -1000, upper_bound: float = 1000) -> pd.DataFrame:
    """
    Apply log transformation to the specified feature columns, and bound the transformed values.
    
    Args:
        df (pd.DataFrame): The dataset containing features to log transform.
        feature_columns (list): List of feature column names to log transform.
        lower_bound (float): Lower bound for the transformed values.
        upper_bound (float): Upper bound for the transformed values.
        
    Returns:
        pd.DataFrame: A DataFrame with the log-transformed and bounded features.
    """
    for col in feature_columns:
        if (df[col] <= 0).any():
            print(f"Skipping log transformation for {col} due to non-positive values.")
            continue
        # Apply log transformation
        df[col + '_log'] = np.log(df[col])
        # Apply bounds
        #df[col + '_log'] = df[col + '_log'].clip(lower=lower_bound, upper=upper_bound)
    
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

def get_preprocessed_df(n_clusters=3, log_transform=False, logFeatures=None, train_features=[], comparison_features=['arat_aff_total', 'fma_ue_aff_total'], comparison_name='cluster_arat_fma'):
    #df_IMU = load_dataset('/cluster/work/vogtlab/Projects/IMUStrokeRecovery/Outcome_files/data_merged_exclGait_THval.csv')
    #df_IMU = load_dataset('/cluster/work/vogtlab/Projects/IMUStrokeRecovery/Outcome_files/data_merged_incl.GAIT_TH2.csv') 
    #df_IMU = load_dataset('~/Notebooks/Data/data_merged_incl.GAIT_TH2_Recalculated.csv')
    df_IMU = load_dataset('~/Notebooks/Data/data_merged_incl.GAIT_TH2_Recalculated_with_Jerk.csv')
    df_fma = load_dataset('/cluster/work/vogtlab/Projects/IMUStrokeRecovery/Outcome_files/REUSE-ClusteringULperforma_DATA_2024-08-02_1038.csv')

    # join datasets
    visit_to_day = { # day 4 -> 28 instead of 5
        2: 'day3',
        3: 'day9',
        4: 'day28',
        6: 'day90',
        7: 'day365'
    }
    
    df_fma['visit'] = df_fma['redcap_event_name'].apply(
        lambda x: visit_to_day.get(int(re.search(r'visit_(\d+)_arm_\d+', x).group(1)), '')
    )
    
    df_fma = df_fma.rename(columns={'id': 'Pat_id'})
    
    df_fma = df_fma[['Pat_id', 'visit', 'fma_ue_aff_total', 'arat_aff_total']]
    
    df = pd.merge(df_IMU, df_fma, on=['Pat_id', 'visit', 'arat_aff_total'])

    columns_to_fill = ['gender', 'dom_aff']

    for column in columns_to_fill:
        df[column] = df.groupby('Pat_id')[column].ffill().bfill()

    df['day_visit'] = df.apply(lambda row: int(row['visit'].replace('day','')) if pd.isna(row['day_visit']) else row['day_visit'], axis=1)

    #df = categorize_arat_fma(df)

    df = calc_alternative_ratios(df)

    #df.dropna(subset=comparison_features, inplace=True)
    #kmeans_model = cluster_data(df, comparison_features, n_clusters)
    
    scaler = StandardScaler()
    scaled_comparison_features = scaler.fit_transform(df[comparison_features])
    comparison_features_scaled = []
    
    for i, feature in enumerate(comparison_features):
        df[f'{feature}_scaled'] = scaled_comparison_features[:, i]
        comparison_features_scaled += [f'{feature}_scaled']

    #df = df.dropna()
    #feature_cols_scaled = ['arat_aff_total_scaled', 'fma_ue_aff_total_scaled']
    df.dropna(subset=comparison_features_scaled, inplace=True)
    kmeans_model = cluster_data(df, comparison_features_scaled, n_clusters)#, day=90)

    df = predict_clusters(df, comparison_name, comparison_features_scaled, kmeans_model)
    df = order_clusters(df, comparison_name, comparison_features)

    df.dropna(subset=train_features, inplace=True)
    if log_transform and logFeatures is not None:
        df = log_transform_features(df, logFeatures)

    #df = df[(df['Pat_id'] != 'RU036') & (df['visit'] != 'day9')] 
    #df = df[(df['Pat_id'] != 'RU062') & (df['visit'] != 'day90')] 
    #print(df[(df['Pat_id'] == 'RU036') & (df['visit'] == 'day9')])
    
    return df