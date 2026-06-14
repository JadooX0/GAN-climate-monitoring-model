import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # Maximizes parallel execution on GPU

NUM_EPOCHS = 800       
BATCH_SIZE = 32          
LR = 0.0002
BETA1 = 0.5
BETA2 = 0.999
LAMBDA_L1 = 100.0       


class CopernicusMultiFileDataset(Dataset):
    """
    Automating multi-file handling for time-split Copernicus datasets.
    Combines independent .nc file chunks seamlessly.
    """
    def __init__(self, file_pattern):
        super(CopernicusMultiFileDataset, self).__init__()
        
        # Find all files matching pattern
        self.file_paths = sorted(glob.glob(file_pattern))
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No NetCDF files found matching pattern: {file_pattern}")
            
        print(f"Successfully found and merging {len(self.file_paths)} files into memory...")
        
        
        with xr.open_mfdataset(self.file_paths, combine='by_coords', chunks={'time': 100}) as ds:
           
            self.temp = ds['t2m'].values          
            self.dew_point = ds['d2m'].values     
            self.ssrd_target = ds['ssrd'].values   
        
        
        self.temp = self._scale_data(self.temp)
        self.dew_point = self._scale_data(self.dew_point)
        self.ssrd_target = self._scale_data(self.ssrd_target)

    def _scale_data(self, matrix):
        min_v, max_v = np.nanmin(matrix), np.nanmax(matrix)
        matrix = np.nan_to_num(matrix, nan=min_v)
        if max_v - min_v == 0:
            return matrix - min_v
        return ((matrix - min_v) / (max_v - min_v)) * 2.0 - 1.0

    def __len__(self):
        return self.ssrd_target.shape[0]

    def __getitem__(self, idx):
        
        input_tensor = np.stack([self.temp[idx], self.dew_point[idx]], axis=0).astype(np.float32)
        
        target_tensor = np.expand_dims(self.ssrd_target[idx], axis=0).astype(np.float32)
        
       
        input_tensor = self._match_dimensions(input_tensor, 256, 256)
        target_tensor = self._match_dimensions(target_tensor, 256, 256)
        
        return torch.tensor(input_tensor), torch.tensor(target_tensor)

    def _match_dimensions(self, tensor, target_h, target_w):
        c, h, w = tensor.shape
        if h == target_h and w == target_w:
            return tensor
        tmp = np.zeros((c, target_h, target_w), dtype=np.float32)
        min_h, min_w = min(h, target_h), min(w, target_w)
        tmp[:, :min_h, :min_w] = tensor[:, :min_h, :min_w]
        return tmp


class DroGenerator(nn.Module):
    def __init__(self):
        super(DroGenerator, self).__init__()
        # 2 Input Channels -> [t2m, d2m]
        self.enc1 = nn.Conv2d(2, 64, kernel_size=4, stride=2, padding=1) 
        self.enc2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128)
        )
        self.dec1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )
        self.dec2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.dec2(self.dec1(self.enc2(self.enc1(x))))

class DroDiscriminator(nn.Module):
    def __init__(self):
        super(DroDiscriminator, self).__init__()
        # Input shape matches condition maps (2 channels) + SSI evaluation target (1 channel) = 3 total channels
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, climate_vars, ssi_map):
        return self.model(torch.cat([climate_vars, ssi_map], dim=1))


netG = DroGenerator().to(device)
netD = DroDiscriminator().to(device)

criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()

optimizerG = optim.Adam(netG.parameters(), lr=LR, betas=(BETA1, BETA2))
optimizerD = optim.Adam(netD.parameters(), lr=LR, betas=(BETA1, BETA2))


def run_pipeline():
    
    file_search_pattern = "*.nc" 
    
    try:
        dataset = CopernicusMultiFileDataset(file_search_pattern)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    except Exception as e:
        print(f"\n[Initialization Error]: {e}")
        print(" -> Fix: Move your two downloaded '.nc' files into this exact folder hierarchy context.\n")
        return

    print(f"Starting training on device: {device}...")
    for epoch in range(1, NUM_EPOCHS + 1):
        for i, (climate_vars, real_ssi) in enumerate(dataloader):
            climate_vars = climate_vars.to(device)
            real_ssi = real_ssi.to(device)
            
            
            patch_dimensions = netD(climate_vars, real_ssi).size()
            real_label = torch.ones(patch_dimensions, device=device)
            fake_label = torch.zeros(patch_dimensions, device=device)
            
            
            optimizerD.zero_grad()
            loss_D_real = criterion_GAN(netD(climate_vars, real_ssi), real_label)
            
            fake_ssi = netG(climate_vars)
            loss_D_fake = criterion_GAN(netD(climate_vars, fake_ssi.detach()), fake_label)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizerD.step()
            
            
            optimizerG.zero_grad()
            loss_G_GAN = criterion_GAN(netD(climate_vars, fake_ssi), real_label)
            loss_G_L1 = criterion_L1(fake_ssi, real_ssi)
            
            loss_G = loss_G_GAN + (LAMBDA_L1 * loss_G_L1)
            loss_G.backward()
            optimizerG.step()
            
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] | Discriminator Loss: {loss_D.item():.4f} | Generator Loss: {loss_G.item():.4f}")

    
    print("Running distribution verification passes...")
    netG.eval()
    val_loader = DataLoader(dataset, batch_size=40, shuffle=False)
    val_climate, val_real_ssi = next(iter(val_loader))
    
    with torch.no_grad():
        val_fake_ssi = netG(val_climate.to(device)).cpu().numpy()
    
    real_flat = val_real_ssi.numpy().reshape(val_real_ssi.shape[0], -1)
    fake_flat = val_fake_ssi.reshape(val_fake_ssi.shape[0], -1)
    combined = np.vstack([real_flat, fake_flat])
    
    pca_output = PCA(n_components=2).fit_transform(combined)
    tsne_output = TSNE(n_components=2, perplexity=10, random_state=42).fit_transform(combined)
    
    split_idx = val_real_ssi.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(pca_output[:split_idx, 0], pca_output[:split_idx, 1], c='royalblue', alpha=0.6, label='Copernicus Real Data')
    axes[0].scatter(pca_output[split_idx:, 0], pca_output[split_idx:, 1], c='crimson', alpha=0.6, label='DroGAN Synthetic Maps')
    axes[0].set_title('PCA Latent Variance Alignment')
    axes[0].legend()
    
    axes[1].scatter(tsne_output[:split_idx, 0], tsne_output[:split_idx, 1], c='royalblue', alpha=0.6, label='Copernicus Real Data')
    axes[1].scatter(tsne_output[split_idx:, 0], tsne_output[split_idx:, 1], c='crimson', alpha=0.6, label='DroGAN Synthetic Maps')
    axes[1].set_title('t-SNE Density Verification')
    axes[1].legend()
    
    plt.savefig("drogan_copernicus_evaluation.png")
    print("Verification metrics exported successfully as 'drogan_copernicus_evaluation.png'.")

if __name__ == "__main__":
    run_pipeline()