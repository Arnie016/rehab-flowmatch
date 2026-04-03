import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x, t_emb):
        # x: [batch, in_channels, time]
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t_emb)
        time_emb = time_emb.unsqueeze(-1)  # [batch, out_channels, 1]
        h = h + time_emb
        
        h = self.conv2(h)
        h = self.bn2(h)
        return self.relu(h)

class TemporalUNet(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        # We expect input x to be shape [batch, time, feature_dim]
        # We'll transpose it to [batch, feature_dim, time] for Conv1d
        
        # Down 1
        self.down1 = TemporalBlock(feature_dim * 2, hidden_dim, time_emb_dim) # feature_dim*2 due to mask concat
        self.pool1 = nn.MaxPool1d(2)
        
        # Down 2
        self.down2 = TemporalBlock(hidden_dim, hidden_dim * 2, time_emb_dim)
        self.pool2 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = TemporalBlock(hidden_dim * 2, hidden_dim * 2, time_emb_dim)
        
        # Up 2
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv2 = TemporalBlock(hidden_dim * 4, hidden_dim, time_emb_dim)
        
        # Up 1
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up_conv1 = TemporalBlock(hidden_dim * 2, hidden_dim, time_emb_dim)
        
        # Output
        self.final_conv = nn.Conv1d(hidden_dim, feature_dim, kernel_size=1)

    def forward(self, x, mask, t):
        # x: [batch, time, feature_dim]
        # mask: [batch, time, feature_dim]
        # Concat x and mask
        h = torch.cat([x, mask.float()], dim=-1)
        h = h.transpose(1, 2)  # [batch, in_channels, time]
        
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encode
        d1 = self.down1(h, t_emb)
        p1 = self.pool1(d1)
        
        d2 = self.down2(p1, t_emb)
        p2 = self.pool2(d2)
        
        # Bottleneck
        b = self.bottleneck(p2, t_emb)
        
        # Decode
        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1) # skip connection
        u2 = self.up_conv2(u2, t_emb)
        
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_conv1(u1, t_emb)
        
        out = self.final_conv(u1)
        out = out.transpose(1, 2) # [batch, time, feature_dim]
        return out
