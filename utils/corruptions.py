import torch
import numpy as np
from typing import Tuple

def add_sensor_noise(x: torch.Tensor, severity: float) -> torch.Tensor:
    """Additive sensor noise."""
    noise_level = severity * 0.1  # Example scaling
    noise = torch.randn_like(x) * noise_level
    return x + noise

def add_drift(x: torch.Tensor, severity: float) -> torch.Tensor:
    """Low-frequency drift (random walk bias)."""
    # x shape: [batch, time, features] or [time, features]
    drift_scale = severity * 0.05
    noise = torch.randn_like(x) * drift_scale
    drift = torch.cumsum(noise, dim=-2)
    return x + drift

def add_dropout(x: torch.Tensor, mask: torch.Tensor, severity: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dropout windows (simulating packet loss)."""
    drop_prob = min(severity * 0.1, 0.5)
    # Generate dropout masks across time
    time_len = x.shape[-2]
    # Simple independent dropout for now
    drop_mask = torch.rand(x.shape[:-1]) > drop_prob
    drop_mask = drop_mask.unsqueeze(-1).expand_as(x)
    new_mask = mask & drop_mask
    return x * new_mask, new_mask

def add_feature_masking(x: torch.Tensor, mask: torch.Tensor, severity: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly mask out specific channels/joints."""
    drop_prob = min(severity * 0.1, 0.6)
    feat_len = x.shape[-1]
    
    # Mask feature channels across the entire window
    feat_mask = torch.rand(x.shape[:-2] + (1, feat_len)) > drop_prob
    feat_mask = feat_mask.expand_as(x)
    new_mask = mask & feat_mask
    return x * new_mask, new_mask

class CorruptionSimulator:
    def __init__(self):
        self.corruption_types = {
            "noise": add_sensor_noise,
            "drift": add_drift,
        }

    def apply(self, x: torch.Tensor, mask: torch.Tensor, c_type: str, severity: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply a parameterized corruption to the input tensor."""
        if c_type == "none" or severity == 0.0:
            return x, mask
            
        if c_type == "dropout":
            return add_dropout(x, mask, severity)
        if c_type == "masking":
            return add_feature_masking(x, mask, severity)
            
        if c_type in self.corruption_types:
            corrupted_x = self.corruption_types[c_type](x, severity)
            return corrupted_x, mask
            
        return x, mask
