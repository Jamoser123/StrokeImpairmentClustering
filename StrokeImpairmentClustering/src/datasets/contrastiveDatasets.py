# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler
from typing import Iterator, List, Tuple, Optional
import numpy as np
import pandas as pd
from src.util.classes import HDFStoreManager
from src.util.funcs import parse_time

# TODO: Add Swap strategy & combine adv and standard train dataset

class FMATrainDataset(Dataset):
    def __init__(self,
                 cfg,
                 days=['day9'],
                 feature_columns=['acc_x', 'acc_y', 'acc_z'],
                 exclude_patients=None,
                 augment=True,
                 norm_params=None):
        """
        A PyTorch Dataset for handling FMA (Fugl-Meyer Assessment) time series data stored in HDF5 format.
        
        Args:
            
            feature_columns (list of str, optional): 
                List of column names to use as features. 
                Default: ['acc_x', 'acc_y', 'acc_z']
            
            exclude_patients (list of str/int, optional): 
                List of patient IDs to exclude from the dataset. 
                Default: None
            
        Returns:
            tuple: (features, fma_score)
                - features: torch.FloatTensor of shape (sequence_length, n_features)
                - fma_score: torch.FloatTensor of shape (1,) containing the FMA score
        """
        
        self.hdf_path = cfg.hdf_path
        self.normalize = cfg.ml.normalize
        self.exclude_patients = exclude_patients if exclude_patients is not None else []
        self.feature_columns = feature_columns
        self.norm_params = {'mean': {}, 'std': {}}
        self.days = days
        self.sample_strat = cfg.ml.sample_strat
        
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

        self.patients_data = {}
        for idx, segment in self.segments.iterrows():
            key = (segment['Pat_id'], segment['day'])
            if key not in self.patients_data:
                self.patients_data[key] = []
            self.patients_data[key].append(idx)

        if self.normalize:
            if norm_params is not None:
                # Use provided parameters (for val/test sets)
                self.norm_params = norm_params
            else:
                # Compute parameters (for training set)
                self._compute_normalization_params()

        self.noise_level = cfg.ml.noise_level
        self.mask_ratio = cfg.ml.mask_ratio
        self.n_segments = cfg.ml.n_shuffleSegments
        self.train_one_wrist = cfg.ml.train_one_wrist

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

    def _augment_transform(self, data, swap):
        seq_len, features = data.shape
        # Random side switching # we need to swap the left side
        # if torch.rand(1).item() < 0.5:#swap:#torch.rand(1).item() < 0.5:  # 50% probability to switch sides
        #     # Features need to be ordered that all features from one side come first then all from the other
        #     # Create a view of the data as [seq_len, 2 (sides), len(features)/2 (features per side)]
        #     data_reshaped = data.view(seq_len, 2, -1)
        #     # Swap the sides (dimension 1)
        #     data_reshaped = torch.flip(data_reshaped, [1])
        #     # Reshape back to original format
        #     data = data_reshaped.reshape(seq_len, -1)
        if self.train_one_wrist:
            if torch.rand(1).item() < 0.5:
                data = data[:, :6]
            else:
                data = data[:, -6:]
        
        # Add noise
        noise = torch.randn_like(data) * self.noise_level
        data = data + noise
    
        # Mask
        mask_length = int(seq_len * self.mask_ratio)
        if mask_length > 0 and (seq_len - mask_length) > 0:
            start_idx = torch.randint(0, seq_len - mask_length, (1,)).item()
            data[start_idx:start_idx + mask_length] = 0
    
        # Slice and shuffle
        seq_len, num_variables = data.shape
        slice_size = seq_len // self.n_segments
        num_slices = self.n_segments
        data = data.transpose(0, 1)
        slices = data[:, :num_slices * slice_size].reshape(num_variables, num_slices, slice_size)
        permuted_indices = torch.randperm(num_slices)
        shuffled_slices = slices[:, permuted_indices, :]
        shuffled_data = shuffled_slices.reshape(num_variables, -1)
        shuffled_data = shuffled_data.transpose(0, 1)
        
        return shuffled_data

    def get_views(self, features1, features2):
        view1 = self._augment_transform(features1.clone(), swap=False)
        view2 = self._augment_transform(features2.clone(), swap=False)
    
        return view1, view2
        
    def __getitem__(self, patient_day_tuple):

        if self.sample_strat == 'self_sample':
            idx = np.random.choice(self.patients_data[patient_day_tuple])
            features, _ = self._load_segment_data(idx)
    
            features = torch.tensor(features, dtype=torch.float)
    
            view1, view2 = self.get_views(features, features)
            return view1, view2, patient_day_tuple
        # Select two random samples for the given patient ID and day
        idx1 = np.random.choice(self.patients_data[patient_day_tuple])
        idx2 = np.random.choice(self.patients_data[patient_day_tuple])
        features1, _ = self._load_segment_data(idx1)
        features2, _ = self._load_segment_data(idx2)
        
        # Convert features to PyTorch tensor immediately after loading
        features1 = torch.tensor(features1, dtype=torch.float)  # Use torch.tensor to convert and specify dtype
        features2 = torch.tensor(features2, dtype=torch.float)

        view1, view2 = self.get_views(features1, features2)
        return view1, view2, patient_day_tuple
    
    def get_patient_ids(self):
        """Return list of unique patient IDs (excluding the filtered ones)"""
        return self.patient_ids

    def get_excluded_patiens(self):
        """Return excluded patient IDS"""
        return self.exclude_patients

def augment_collate_fn(batch):
    """Custom collate function to handle batches of data."""
    features1, features2, labels = zip(*batch)
    return torch.stack(features1), torch.stack(features2), labels

class DistributedPatientDaySampler(BatchSampler):
    def __init__(
        self, 
        dataset: Dataset, 
        batch_size: int,
        num_batches: Optional[int] = None,
        seed: int = 42,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.patient_days = list(dataset.patients_data.keys())
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        
        # Store total batches calculation for later
        if num_batches is not None:
            self.total_batches = num_batches
        else:
            self.total_batches = len(dataset) // batch_size
            
    def __iter__(self) -> Iterator[List[Tuple]]:
        # Get distributed info in __iter__ when environment is initialized
        if torch.distributed.is_initialized():
            self.num_replicas = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.num_replicas = 1
            self.rank = 0
            
        # Calculate num_batches here
        self.num_batches = self.total_batches // self.num_replicas
        if not self.drop_last and self.total_batches % self.num_replicas != 0:
            self.num_batches += 1
            
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + self.rank)
        
        for _ in range(self.num_batches):
            batch_pairs = []
            remaining = self.batch_size
            
            while remaining > 0:
                indices = torch.randperm(len(self.patient_days), generator=generator).tolist()
                shuffled_pairs = [self.patient_days[i] for i in indices]
                
                samples_to_take = min(remaining, len(shuffled_pairs))
                batch_pairs.extend(shuffled_pairs[:samples_to_take])
                remaining -= samples_to_take
            
            yield batch_pairs
    
    def __len__(self) -> int:
        # Need to handle this carefully since num_batches isn't set until __iter__
        if not hasattr(self, 'num_batches'):
            if torch.distributed.is_initialized():
                num_replicas = torch.distributed.get_world_size()
            else:
                num_replicas = 1
            return self.total_batches // num_replicas
        return self.num_batches

class AugmentPatientDayDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, **kwargs):
        super().__init__(dataset, batch_sampler=DistributedPatientDaySampler(dataset, batch_size), collate_fn=augment_collate_fn, **kwargs)

def create_traindatasets(cfg, train_patients, val_patients, feature_columns):
    """Create train and validation datasets"""
    train_dataset = FMATrainDataset(
        cfg=cfg,
        exclude_patients=val_patients,  # Exclude validation patients
        feature_columns=feature_columns,
        days=cfg.ml.train_days,
    )
    norm_params = train_dataset.get_norm_params()
    
    val_dataset = FMATrainDataset(
        cfg=cfg,
        exclude_patients=train_patients,  # Exclude training patients
        feature_columns=feature_columns,
        days=cfg.ml.train_days,
        norm_params=norm_params
    )
    
    return train_dataset, val_dataset

def create_traindataloaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """Create train and validation dataloaders"""
    train_loader = AugmentPatientDayDataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = AugmentPatientDayDataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader


############################# Raw Dataset for embeddings
class FMAEmbeddingDataset(Dataset):
    def __init__(self,
                 cfg,
                 days=['day9'],
                 feature_columns=['acc_x', 'acc_y', 'acc_z'],
                 exclude_patients=None,
                 augment=True,
                 norm_params=None):
        """
        A PyTorch Dataset for handling FMA (Fugl-Meyer Assessment) time series data stored in HDF5 format.
        
        Args:
            
            feature_columns (list of str, optional): 
                List of column names to use as features. 
                Default: ['acc_x', 'acc_y', 'acc_z']
            
            exclude_patients (list of str/int, optional): 
                List of patient IDs to exclude from the dataset. 
                Default: None
            
        Returns:
            tuple: (features, fma_score)
                - features: torch.FloatTensor of shape (sequence_length, n_features)
                - fma_score: torch.FloatTensor of shape (1,) containing the FMA score
        """
        
        self.hdf_path = cfg.hdf_path
        self.normalize = cfg.ml.normalize
        self.exclude_patients = exclude_patients if exclude_patients is not None else []
        self.feature_columns = feature_columns
        self.norm_params = {'mean': {}, 'std': {}}
        self.days = days
        
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

        self.patients_data = {}
        for idx, segment in self.segments.iterrows():
            key = (segment['Pat_id'], segment['day'])
            if key not in self.patients_data:
                self.patients_data[key] = []
            self.patients_data[key].append(idx)

        if self.normalize:
            if norm_params is not None:
                # Use provided parameters (for val/test sets)
                self.norm_params = norm_params
            else:
                # Compute parameters (for training set)
                self._compute_normalization_params()

        self.noise_level = cfg.ml.noise_level
        self.mask_ratio = cfg.ml.mask_ratio
        self.n_segments = cfg.ml.n_shuffleSegments
        self.train_one_wrist = cfg.ml.train_one_wrist

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

    def _augment_transform(self, data):
        seq_len, features = data.shape
        # Random side switching
        if torch.rand(1).item() < 0.5:  # 50% probability to switch sides
            # Features need to be ordered that all features from one side come first then all from the other
            # Create a view of the data as [seq_len, 2 (sides), len(features)/2 (features per side)]
            data_reshaped = data.view(seq_len, 2, -1)
            # Swap the sides (dimension 1)
            data_reshaped = torch.flip(data_reshaped, [1])
            # Reshape back to original format
            data = data_reshaped.reshape(seq_len, -1)
        return data
        
    def __getitem__(self, idx):
        features, key = self._load_segment_data(idx)
        features = torch.tensor(features, dtype=torch.float)
        #features = self._augment_transform(features.clone())
        if self.train_one_wrist:
            features = features[:, :6] # get only the 
        
        return features, key
        
    def get_patient_ids(self):
        """Return list of unique patient IDs (excluding the filtered ones)"""
        return self.patient_ids

    def get_excluded_patiens(self):
        """Return excluded patient IDS"""
        return self.exclude_patients

def default_collate_fn(batch):
    """Custom collate function without augmentation."""
    features, keys = zip(*batch)
    return torch.stack(features), keys

def create_embeddingDataset(cfg, feature_columns):
    """Create embedding Dataset"""
    dataset = FMAEmbeddingDataset(
        cfg=cfg,
        feature_columns=feature_columns,
        days=cfg.ml.train_days
    )
    return dataset

def create_embeddingLoader(cfg, dataset):
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.ml.batch_size,
        num_workers=cfg.ml.num_workers,
        collate_fn=default_collate_fn,
        pin_memory=True
    )
    return loader

#### Dataset + functions for adversial model
class FMAADVDataset(Dataset): # We can optimise this such that we just need one training dataset with additional configs
    def __init__(self,
                 cfg,
                 days=['day9'],
                 feature_columns=['acc_x', 'acc_y', 'acc_z'],
                 exclude_patients=None,
                 augment=True,
                 norm_params=None):
        """
        A PyTorch Dataset for handling FMA (Fugl-Meyer Assessment) time series data stored in HDF5 format.
        
        Args:
            
            feature_columns (list of str, optional): 
                List of column names to use as features. 
                Default: ['acc_x', 'acc_y', 'acc_z']
            
            exclude_patients (list of str/int, optional): 
                List of patient IDs to exclude from the dataset. 
                Default: None
            
        Returns:
            tuple: (features, fma_score)
                - features: torch.FloatTensor of shape (sequence_length, n_features)
                - fma_score: torch.FloatTensor of shape (1,) containing the FMA score
        """
        
        self.hdf_path = cfg.hdf_path
        self.normalize = cfg.ml.normalize
        self.exclude_patients = exclude_patients if exclude_patients is not None else []
        self.feature_columns = feature_columns
        self.norm_params = {'mean': {}, 'std': {}}
        self.days = days
        self.sample_strat = cfg.ml.sample_strat
        
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

        self.patients_data = {}
        for idx, segment in self.segments.iterrows():
            key = (segment['Pat_id'], segment['day'])
            if key not in self.patients_data:
                self.patients_data[key] = []
            self.patients_data[key].append(idx)

        if self.normalize:
            if norm_params is not None:
                # Use provided parameters (for val/test sets)
                self.norm_params = norm_params
            else:
                # Compute parameters (for training set)
                self._compute_normalization_params()

        self.noise_level = cfg.ml.noise_level
        self.mask_ratio = cfg.ml.mask_ratio
        self.n_segments = cfg.ml.n_shuffleSegments

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
            
            return features, segment['aff_side']

    def __len__(self):
        return len(self.segments)

    def _augment_transform(self, data, swap):
        seq_len, features = data.shape
        # Random side switching
        if swap:#torch.rand(1).item() < 0.5:  # 50% probability to switch sides
            # Features need to be ordered that all features from one side come first then all from the other
            # Create a view of the data as [seq_len, 2 (sides), len(features)/2 (features per side)]
            data_reshaped = data.view(seq_len, 2, -1)
            # Swap the sides (dimension 1)
            data_reshaped = torch.flip(data_reshaped, [1])
            # Reshape back to original format
            data = data_reshaped.reshape(seq_len, -1)
    
        # Add noise
        noise = torch.randn_like(data) * self.noise_level
        data = data + noise
    
        # Mask
        mask_length = int(seq_len * self.mask_ratio)
        if mask_length > 0 and (seq_len - mask_length) > 0:
            start_idx = torch.randint(0, seq_len - mask_length, (1,)).item()
            data[start_idx:start_idx + mask_length] = 0
    
        # Slice and shuffle
        seq_len, num_variables = data.shape
        slice_size = seq_len // self.n_segments
        num_slices = self.n_segments
        data = data.transpose(0, 1)
        slices = data[:, :num_slices * slice_size].reshape(num_variables, num_slices, slice_size)
        permuted_indices = torch.randperm(num_slices)
        shuffled_slices = slices[:, permuted_indices, :]
        shuffled_data = shuffled_slices.reshape(num_variables, -1)
        shuffled_data = shuffled_data.transpose(0, 1)
        
        return shuffled_data

    def get_views(self, features1, features2):
        view1 = self._augment_transform(features1.clone(), swap=False)
        view2 = self._augment_transform(features2.clone(), swap=True)
    
        return view1, view2
        
    def __getitem__(self, patient_day_tuple):

        if self.sample_strat == 'self_sample':
            idx = np.random.choice(self.patients_data[patient_day_tuple])
            features, aff_side = self._load_segment_data(idx)
    
            features = torch.tensor(features, dtype=torch.float)
    
            view1, view2 = self.get_views(features, features)
            return view1, view2, aff_side, patient_day_tuple
        # Select two random samples for the given patient ID and day
        idx1 = np.random.choice(self.patients_data[patient_day_tuple])
        idx2 = np.random.choice(self.patients_data[patient_day_tuple])
        features1, aff_side = self._load_segment_data(idx1) #aff_side is the same in both patients
        features2, aff_side2 = self._load_segment_data(idx2)

        if aff_side != aff_side2:
            print(f"Two samples from same patient with different affected_sides encountered! Patient: {patient_day_tuple} idx1: {idx1} idx2: {idx2}") 
        # for debug purposes if aff_side1 != aff_side2: print("ERROR")
        
        # Convert features to PyTorch tensor immediately after loading
        features1 = torch.tensor(features1, dtype=torch.float)  # Use torch.tensor to convert and specify dtype
        features2 = torch.tensor(features2, dtype=torch.float)

        view1, view2 = self.get_views(features1, features2)
        return view1, view2, aff_side, patient_day_tuple
    
    def get_patient_ids(self):
        """Return list of unique patient IDs (excluding the filtered ones)"""
        return self.patient_ids

    def get_excluded_patiens(self):
        """Return excluded patient IDS"""
        return self.exclude_patients

def adv_collate_fn(batch):
    """Custom collate function to handle batches of data."""
    features1, features2, aff_sides, labels = zip(*batch)
    return torch.stack(features1), torch.stack(features2), torch.tensor(aff_sides), labels


class ADVPatientDayDataLoader(DataLoader):
    def __init__(self, dataset, batch_size=64, **kwargs):
        super().__init__(dataset, batch_sampler=DistributedPatientDaySampler(dataset, batch_size), collate_fn=adv_collate_fn, **kwargs)

def create_advDatasets(cfg, train_patients, val_patients, feature_columns):
    """Create train and validation datasets"""
    train_dataset = FMAADVDataset(
        cfg=cfg,
        exclude_patients=val_patients,  # Exclude validation patients
        feature_columns=feature_columns,
        days=cfg.ml.train_days,
    )
    norm_params = train_dataset.get_norm_params()
    
    val_dataset = FMAADVDataset(
        cfg=cfg,
        exclude_patients=train_patients,  # Exclude training patients
        feature_columns=feature_columns,
        days=cfg.ml.train_days,
        norm_params=norm_params
    )
    
    return train_dataset, val_dataset

def create_advLoaders(train_dataset, val_dataset, batch_size=32, num_workers=4):
    """Create train and validation dataloaders"""
    train_loader = ADVPatientDayDataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = ADVPatientDayDataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
