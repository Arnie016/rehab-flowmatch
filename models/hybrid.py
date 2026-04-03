import torch
import torch.nn as nn

class StatefulLIFCell(nn.Module):
    """
    A simple Leaky Integrate-and-Fire neuron for the event-driven SNN residual predictor.
    """
    def __init__(self, in_features, beta=0.9, threshold=1.0):
        super().__init__()
        self.fc = nn.Linear(in_features, in_features)
        self.beta = beta
        self.threshold = threshold

    def forward(self, x, membrane_state):
        # Inject current
        synaptic_current = self.fc(x)
        
        # Leaky integrate
        membrane_state = (self.beta * membrane_state) + synaptic_current
        
        # Fire spikes if membrane exceeds threshold
        spikes = (membrane_state >= self.threshold).float()
        
        # Reset by subtraction
        membrane_state = membrane_state - (spikes * self.threshold)
        
        return spikes, membrane_state


class ANNStudent(nn.Module):
    """
    Compact Temporal Backbone trained via distillation from the FlowMatch teacher.
    It produces coarse low-frequency global trajectory reconstructions.
    """
    def __init__(self, in_channels, out_channels, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=hidden_dim), # Depthwise
            nn.ReLU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1) # Pointwise
        )

    def forward(self, x):
        # x: [batch, time, features] -> [batch, features, time]
        h = x.transpose(1, 2)
        out = self.net(h)
        return out.transpose(1, 2)


class SNNResidual(nn.Module):
    """
    Event-driven high-rate residual predictor acting on frame-to-frame incremental changes.
    """
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        # Simplified continuous-time recurrent spiking residual block
        self.lif_encoder = StatefulLIFCell(feature_dim)
        self.recurrent_lif = StatefulLIFCell(feature_dim)
        
        # Decoder weights mapping spikes back to continuous residuals
        self.spike_decoder = nn.Linear(feature_dim, feature_dim, bias=False)

    def forward(self, delta_x, init_membrane=None):
        batch, time_steps, features = delta_x.shape
        device = delta_x.device
        
        if init_membrane is None:
            mem_enc = torch.zeros(batch, features, device=device)
            mem_rec = torch.zeros(batch, features, device=device)
        else:
            mem_enc, mem_rec = init_membrane
            
        residual_trajectory = []
        
        # Process incrementally step-by-step through time
        for t in range(time_steps):
            x_t = delta_x[:, t, :]
            
            # Layer 1 Spikes
            spk_enc, mem_enc = self.lif_encoder(x_t, mem_enc)
            
            # Layer 2 Spikes
            spk_rec, mem_rec = self.recurrent_lif(spk_enc, mem_rec)
            
            # Decode spikes into continuous residual correction for this frame
            res_t = self.spike_decoder(spk_rec)
            residual_trajectory.append(res_t.unsqueeze(1))
            
        return torch.cat(residual_trajectory, dim=1), (mem_enc, mem_rec)


class HybridANNSNN(nn.Module):
    """
    Combines the low-power ANN student (for structural low-frequency frame generation) 
    with the SNN residual event-corrector (for high-frequency stability tuning).
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.ann_student = ANNStudent(in_channels=feature_dim * 2, out_channels=feature_dim)
        self.snn_residual = SNNResidual(feature_dim)

    def forward(self, x_corrupted, mask):
        # 1. Base prediction from efficient ANN
        # Concatenate mask as in standard formulation
        ann_input = torch.cat([x_corrupted, mask.float()], dim=-1)
        coarse_traj = self.ann_student(ann_input)
        
        # 2. Extract frame-to-frame incremental changes (deltas) of the coarse trajectory
        # Prepend zero for the first difference
        shifted_coarse = torch.cat([torch.zeros_like(coarse_traj[:, :1, :]), coarse_traj[:, :-1, :]], dim=1)
        frame_deltas = coarse_traj - shifted_coarse
        
        # 3. Predict high-rate SNN residual corrections
        fine_residuals, _ = self.snn_residual(frame_deltas)
        
        # 4. Corrected Hybrid output
        return coarse_traj + fine_residuals
