#classifierDataset.py
class FMADataset(Dataset):
    def __init__(self,
                 cfg,
                 days=['day9'],
                 feature_columns=['acc_x', 'acc_y', 'acc_z'],
                 transform=None, 
                 exclude_patients=None,
                 norm_params=None):
        """
        A PyTorch Dataset for handling FMA (Fugl-Meyer Assessment) time series data stored in HDF5 format.
        
        Args:
            
            feature_columns (list of str, optional): 
                List of column names to use as features. 
                Default: ['acc_x', 'acc_y', 'acc_z']
            
            days (list of str, optional): 
                List of days to include in the dataset. Each day should be in format 'dayX' 
                where X is the day number. Default: ['day9']
            
            transform (callable, optional): 
                Optional transform to be applied on each sample.
                Default: None
            
            exclude_patients (list of str/int, optional): 
                List of patient IDs to exclude from the dataset. 
                Default: None
            
            norm_params (dict, optional):
                Dictionary containing pre-computed normalization parameters. Should have format:
                {'mean': {feature_name: mean_value, ...}, 
                 'std': {feature_name: std_value, ...}}
                If provided, these parameters will be used instead of computing new ones.
                Used for validation/test sets to ensure consistent normalization.
                Default: None
        
        Returns:
            tuple: (features, fma_score)
                - features: torch.FloatTensor of shape (sequence_length, n_features)
                - fma_score: torch.FloatTensor of shape (1,) containing the FMA score
        """
        
        if cfg.ml.do_classification and cfg.ml.num_categories not in [3, 5]:
            raise ValueError("num_categories must be either 3 or 5")
            
        self.use_categories = cfg.ml.do_classification
        self.num_categories = cfg.ml.num_categories
        
        # Category ranges
        if self.use_categories:
            if self.num_categories == 3: # Kmeans categories
                self.category_ranges = {
                    0: (0, 23),    
                    1: (23, 46),   
                    2: (46, 67)   
                }
            else:  # Hohenhoorst Categories
                self.category_ranges = {
                    0: (0, 22),    
                    1: (22, 31),   
                    2: (31, 47),   
                    3: (47, 52),   
                    4: (52, 67)   
                }
        self.hdf_path = cfg.hdf_path
        self.transform = transform
        self.normalize = cfg.ml.normalize
        self.exclude_patients = exclude_patients if exclude_patients is not None else []
        self.days = days
        self.feature_columns = feature_columns
        self.norm_params = {'mean': {}, 'std': {}}

        window_m = int(parse_time(cfg.window_size, 'm'))
        
        # Load segments from all specified days
        segments_list = []
        with HDFStoreManager.get_store(self.hdf_path) as hdf:
            for day in self.days:
                slice_key = f'/slices/window_{window_m}m__threshold_{cfg.threshold_ratio}/{day}'
                day_segments = hdf.get(slice_key)
                day_segments['day'] = day
                segments_list.append(day_segments)
        
        # Concatenate all segments
        self.segments = pd.concat(segments_list, axis=0, ignore_index=True)
        
        # Filter out excluded patients
        if self.exclude_patients:
            self.segments = self.segments[~self.segments['Pat_id'].isin(self.exclude_patients)]
            self.segments = self.segments.reset_index(drop=True)
            
        self.patient_ids = self.segments['Pat_id'].unique()

        if self.normalize:
            if norm_params is not None:
                # Use provided parameters (for val/test sets)
                self.norm_params = norm_params
            else:
                # Compute parameters (for training set)
                self._compute_normalization_params()

    def get_norm_params(self):
        """Return normalization parameters"""
        if self.normalize:
            return self.norm_params
        return None
        
    def _compute_normalization_params(self):
        """Compute mean and standard deviation for each feature across all samples"""
        # Initialize accumulators for online computation
        n_samples = 0
        feature_sums = {col: 0.0 for col in self.feature_columns}
        feature_sq_sums = {col: 0.0 for col in self.feature_columns}

        # First pass: compute means
        with HDFStoreManager.get_store(self.hdf_path) as hdf:
            for idx in range(len(self.segments)):
                segment = self.segments.iloc[idx]
                segment_data = hdf.select(
                    segment['key'],
                    start=segment['start'],
                    stop=segment['end']
                )[self.feature_columns]
                
                for col in self.feature_columns:
                    feature_sums[col] += segment_data[col].sum()
                    feature_sq_sums[col] += (segment_data[col] ** 2).sum()
                n_samples += len(segment_data)

        # Compute means and standard deviations
        for col in self.feature_columns:
            self.norm_params['mean'][col] = feature_sums[col] / n_samples
            mean_sq = feature_sq_sums[col] / n_samples
            variance = mean_sq - (self.norm_params['mean'][col] ** 2)
            self.norm_params['std'][col] = np.sqrt(variance)
    
    def _load_segment_data(self, idx):
        """Load the actual time series data for a segment using efficient selection"""
        segment = self.segments.iloc[idx]
        
        with HDFStoreManager.get_store(self.hdf_path) as hdf:
            # Get only the required segment using select
            segment_data = hdf.select(
                segment['key'],
                start=segment['start'],
                stop=segment['end'],
            )
            segment_data = segment_data[self.feature_columns]
            # Convert to numpy array
            features = segment_data.values
            
            if self.normalize:
                # Apply normalization per feature
                for i, col in enumerate(self.feature_columns):
                    if col in self.norm_params['mean']:
                        features[:, i] = (features[:, i] - self.norm_params['mean'][col]) / self.norm_params['std'][col]
            
            return features, segment['key']

    def __len__(self):
        return len(self.segments)

    def _score_to_category(self, score):
        """Convert FMA score to category"""
        for category, (lower, upper) in self.category_ranges.items():
            if lower <= score < upper:
                return category
        return 0  # Default category for any unexpected values        

    def __getitem__(self, idx): # add the demographic data here
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Load the segment data
        features, key = self._load_segment_data(idx)

        # Get the demographic data
        # Get the demographic data
        row = self.segments.iloc[idx]
        dem = [row['aff_side'], row['gender'], row['age'], row['dom_aff']]
        dem = torch.LongTensor(dem)

        # Get the FMA score
        fma_score = self.segments.iloc[idx]['fma_score']
        
        # Convert to torch tensors
        features = torch.FloatTensor(features)
        
        if self.use_categories:
            # Convert score to category and use one-hot encoding
            category = self._score_to_category(fma_score)
            fma_score = torch.LongTensor([category])
        else:
            fma_score = torch.FloatTensor([fma_score])
        
        if self.transform:
            features = self.transform(features)
            
        return key, features, dem, fma_score
    
    def get_patient_ids(self):
        """Return list of unique patient IDs (excluding the filtered ones)"""
        return self.patient_ids

    def get_excluded_patiens(self):
        """Return excluded patient IDS"""
        return self.exclude_patients
    
    def get_days(self):
        """Return list of days in the dataset"""
        return self.days
    
    def get_day_distribution(self):
        """Return distribution of samples across days"""
        return self.segments['day'].value_counts()
    
    def get_feature_columns(self):
        """Return list of feature columns being used"""
        return self.feature_columns