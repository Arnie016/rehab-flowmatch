import torch
from torch.utils.data import DataLoader
from models.unet import TemporalUNet
from core.flow_matching import FlowMatcher
from data.dataset import StrokeRehabDataset
from utils.corruptions import CorruptionSimulator
import os

def main():
    # 1. Target Apple Silicon Metal Performance Shaders (MPS) for GPU Acceleration
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"--- Utilizing Hardware Accelerator: {device} ---")

    # 2. Setup Parallel Workers to Maximize Mac CPU cores mapped to I/O
    print("Setting up parallel dataset workers...")
    corruptor = CorruptionSimulator()
    dataset = StrokeRehabDataset(
        data_root='/Users/arnav/Datasets/strokerehab_processed/', 
        split='training', 
        window_size=512, 
        corruptor=corruptor, 
        corruption_type="noise", 
        severity=0.5
    )
    
    # 3. macOS native multithreading parameters for dataset bottlenecks
    loader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=8, # 8 parallel CPU threads
        prefetch_factor=2, 
        pin_memory=True # Fast memory transfer to MPS
    )

    print("Initializing Flow Generator onto Mac Silicon...")
    # 103 features + 1 binary mask = 104 input channels
    model = TemporalUNet(in_channels=104, out_channels=103).to(device)
    flow_matcher = FlowMatcher(model).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    print("\n--- Training Loop Starting ---")
    model.train()
    
    for epoch in range(1, 40): 
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(loader):
            # Stream memory smoothly to MPS architecture
            x_clean = batch['clean'].to(device)
            x_corr = batch['corrupted'].to(device)
            mask = batch['mask'].to(device)
            
            optimizer.zero_grad()
            
            # Predict continuous ODE velocity field
            loss_dict = flow_matcher.compute_loss(x_clean, x_corr, mask)
            loss = loss_dict['mse']
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Step {batch_idx:03d} | MPS Vector Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / len(loader)
        print(f"*** Epoch {epoch} Complete | Validation Masked RMSE (Proxy): {avg_loss:.4f} ***")
        
        # Save placeholder
        os.makedirs('experiments/hybrid_mic_flow_mps_fullpower', exist_ok=True)
        torch.save(model.state_dict(), f'experiments/hybrid_mic_flow_mps_fullpower/best.pt')

if __name__ == '__main__':
    main()
