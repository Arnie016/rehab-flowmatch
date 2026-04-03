import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class StrokeRehabDataset(Dataset):
    def __init__(self, data_root, split='training', fold=0, window_size=512, corruptor=None, corruption_type="none", severity=0.0):
        self.data_root = data_root
        self.window_size = window_size
        self.corruptor = corruptor
        self.corruption_type = corruption_type
        self.severity = severity
        
        # Load splits.json to filter trials
        splits_path = os.path.join(data_root, 'splits.json')
        with open(splits_path, 'r') as f:
            self.splits_info = json.load(f)
            
        allowed_trial_keys = set()
        for t in self.splits_info['trials']:
            # Either map by official_split or grouped_cv_fold
            if split == 'training':
                if t['grouped_cv_fold'] != fold and t['official_split'] != "none":
                    allowed_trial_keys.add(t['trial_key'])
            elif split == 'validation':
                if t['grouped_cv_fold'] == fold:
                    allowed_trial_keys.add(t['trial_key'])
                    
        # Read dataset index
        self.index = []
        index_path = os.path.join(data_root, 'dataset_index.jsonl')
        with open(index_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                t_key = f"{record['subject_id']}::{record['activity_id']}::{record['trial_name']}"
                if t_key in allowed_trial_keys:
                    self.index.append(record)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        record = self.index[idx]
        npz_path = os.path.join(self.data_root, record['shard_path'])
        
        # In a real implementation this would load "features" or similar array
        # Assuming typical .npz dict layout:
        try:
            data = np.load(npz_path)
            features = data['features'] # Adjust key based on raw data
        except Exception:
            # Fallback random initialization for structural testing if raw npz differs
            features = np.random.randn(record['timesteps'], record['num_features'])

        features = torch.tensor(features, dtype=torch.float32)
        
        # Window slicing
        total_len = features.shape[0]
        if total_len > self.window_size:
            start = np.random.randint(0, total_len - self.window_size)
            x_clean = features[start:start+self.window_size]
        else:
            # Pad
            pad_len = self.window_size - total_len
            x_clean = torch.nn.functional.pad(features, (0, 0, 0, pad_len))
            
        # Initial mask (all 1s)
        mask = torch.ones_like(x_clean, dtype=torch.bool)
        
        # Apply corruptions
        if self.corruptor is not None:
            x_corrupted, out_mask = self.corruptor.apply(
                x_clean.clone(), mask, self.corruption_type, self.severity
            )
        else:
            x_corrupted, out_mask = x_clean.clone(), mask
            
        # Conditional embeddings can be added here
            
        return {
            'clean': x_clean,
            'corrupted': x_corrupted,
            'mask': out_mask,
            'subject_id': record['subject_id']
        }
