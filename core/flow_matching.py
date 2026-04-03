import torch
import torch.nn as nn

class FlowMatcher:
    def __init__(self, model):
        self.model = model

    def compute_loss(self, x_clean, x_corrupted, mask, t=None):
        """
        Computes the flow matching regression loss.
        x_clean: Target clean trajectories
        x_corrupted: Source corrupted trajectories
        mask: Observation mask
        t: Optional predefined time values, else sampled uniformly
        """
        batch_size = x_clean.shape[0]
        
        if t is None:
            t = torch.rand(batch_size, device=x_clean.device)
            
        # Reshape t for broadcasting
        t_expand = t.view(-1, 1, 1)
        
        # Linear interpolation path between corrupted and clean
        # x_t = t * x_clean + (1 - t) * x_corrupted
        x_t = (1 - t_expand) * x_corrupted + t_expand * x_clean
        
        # Target velocity field is the straight line direction
        v_target = x_clean - x_corrupted
        
        # Model predicts velocity from current state
        v_pred = self.model(x_t, mask, t)
        
        # MSE Regression loss
        loss = nn.functional.mse_loss(v_pred, v_target)
        return loss

    def sample(self, x_corrupted, mask, num_steps=10):
        """
        Euler integration sampling from corrupted towards clean.
        """
        batch_size = x_corrupted.shape[0]
        device = x_corrupted.device
        
        x_t = x_corrupted.clone()
        dt = 1.0 / num_steps
        
        # Fixed-step Euler Integration
        for step in range(num_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            with torch.no_grad():
                v_pred = self.model(x_t, mask, t)
            
            # Update step
            x_t = x_t + v_pred * dt
            
            # Constraint-aware projection / clamping can be added here
            
        return x_t
