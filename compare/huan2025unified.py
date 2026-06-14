# Strict Adaptation of temp2.py Logic using temp1.py Infrastructure
# -----------------------------------------------------------------
# OBJECTIVE:
# Match the algorithmic flow and structure of temp2.py exactly.
# Use timm (resnet10t) and DataParallel from temp1.py.
#
# KEY FIXES:
# 1. Restored 'MLPClassifier' (Backbone Head) matching temp2.py.
# 2. Restored 'BYOLProjectionHead' (Internal BYOL logic) matching byol_pytorch.
# 3. Backbone has MLPClassifier attached (like temp2.py's resnet.fc).
# 4. BYOL extracts features from global pool (ignore classifier for SSL).
# -----------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.amp as amp
from torch.utils.data import DataLoader, Dataset
import random
import numpy as np
import kornia.augmentation as K
import kornia.losses
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
import timm
import copy

# --- Reproducibility ---
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- 1. Dataset Class ---
class PlantDocDataset(Dataset):
    def __init__(self, folder, target_size=(384, 384)):
        valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}
        self.image_paths = []
        if os.path.exists(folder):
            self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) 
                               if os.path.splitext(f)[1].lower() in valid_exts]
        else:
            print(f"Warning: Folder {folder} not found.")

        self.target_size = target_size
        print(f"Found {len(self.image_paths)} images in '{folder}'.")

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert('RGB')
            image = transforms.Resize(self.target_size)(image)
            return transforms.ToTensor()(image)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.image_paths))

# --- 2. Augmentations ---
class GPUAugment(nn.Module):
    def __init__(self, size=384):
        super().__init__()
        self.augs = nn.Sequential(
            K.RandomResizedCrop((size, size), scale=(0.6, 1.0)),
            K.RandomHorizontalFlip(),
            K.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((3, 3), sigma=(0.1, 2.0), p=0.3)
        )
        self.normalize = K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    def forward(self, x): return self.normalize(self.augs(x))

# --- 3. Models and BYOL Wrapper ---

# A. MLPClassifier (From temp2.py - meant for the Backbone's Head)
class MLPClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(x)

# B. Internal BYOL MLP (For Projection/Prediction Heads)
def BYOLProjectionHead(dim, projection_size=256, hidden_size=2048):
    return nn.Sequential(
        nn.Linear(dim, hidden_size),
        nn.BatchNorm1d(hidden_size),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_size, projection_size)
    )

# C. ResNet Wrapper
# Matches behavior of replacing resnet.fc and extracting from avgpool
class ResNetWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        for name, module in self.model.named_children():
            if name == 'fc':
                continue
            x = module(x)
        return torch.flatten(x, 1)

# D. BYOL Custom (Logic matched to byol_pytorch)
class BYOL_Custom(nn.Module):
    def __init__(self, net, augment_fn, projection_size=256, projection_hidden_size=2048, moving_average_decay=0.99):
        super().__init__()
        self.online_encoder = net
        self.augment_fn = augment_fn 
        self.ma_decay = moving_average_decay
        
        # Auto-detect feature dimension
        with torch.no_grad():
            dummy = torch.randn(2, 3, 384, 384)
            # Run through wrapper to get dim
            feature_dim = net(dummy).shape[1]

        # Init BYOL Heads using the correct helper
        self.online_projector = BYOLProjectionHead(feature_dim, projection_size, projection_hidden_size)
        self.online_predictor = BYOLProjectionHead(projection_size, projection_size, projection_hidden_size)

        self.target_encoder = copy.deepcopy(net)
        self.target_projector = copy.deepcopy(self.online_projector)

        for p in self.target_encoder.parameters(): p.requires_grad = False
        for p in self.target_projector.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        for op, tp in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            tp.data = tp.data * self.ma_decay + op.data * (1 - self.ma_decay)
        for op, tp in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            tp.data = tp.data * self.ma_decay + op.data * (1 - self.ma_decay)

    def regression_loss(self, x, y):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        return (2 - 2 * (x * y).sum(dim=-1)).mean()

    def forward(self, x, return_projection=False):
        if return_projection:
            # Assume x is already augmented
            feat = self.online_encoder(x)
            z = self.online_projector(feat)
            return z
        else:
            # BYOL Loss -> Internal Augmentation
            v1 = self.augment_fn(x)
            v2 = self.augment_fn(x)
            
            # Online
            z1 = self.online_projector(self.online_encoder(v1))
            z2 = self.online_projector(self.online_encoder(v2))
            
            online_p1 = self.online_predictor(z1)
            online_p2 = self.online_predictor(z2)
            
            # Target
            with torch.no_grad():
                target_z1 = self.target_projector(self.target_encoder(v1))
                target_z2 = self.target_projector(self.target_encoder(v2))
            
            loss = 0.5 * (self.regression_loss(online_p1, target_z2) + self.regression_loss(online_p2, target_z1))
            return loss

# --- 4. Losses ---

class MIMLoss(nn.Module):
    def __init__(self, min_mask=0.3, max_mask=0.6, alpha=0.5):
        super().__init__()
        self.min_mask = min_mask
        self.max_mask = max_mask
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()
        self.ssim_loss = kornia.losses.SSIMLoss(window_size=11)
        self.normalize = K.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, x, target):
        batch_size, channels, height, width = x.shape
        # Ensure mask generation matches device
        mask_ratio = random.uniform(self.min_mask, self.max_mask)
        mask = torch.rand(batch_size, height, width, device=x.device) < mask_ratio
        mask = mask.unsqueeze(1).expand(-1, channels, -1, -1)

        x_masked = x.clone()
        x_masked[mask] = 0

        x_masked = self.normalize(x_masked)
        target = self.normalize(target)

        mse = self.mse_loss(x_masked, target)
        ssim = self.ssim_loss(x_masked, target)
        return self.alpha * mse + (1 - self.alpha) * (1 - ssim)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        logits = torch.mm(z_i, z_j.T) / self.temperature
        labels = torch.arange(batch_size).to(z_i.device)
        return F.cross_entropy(logits, labels)

# --- 5. Initialization ---

plant_folder = "/kaggle/input/unlabel-full-plantdoc/unlabel"
dataset = PlantDocDataset(plant_folder)
batch_size = 16

train_loader = DataLoader(
    dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=4, 
    drop_last=True, 
    persistent_workers=True
)

# Components
gpu_augment = GPUAugment(size=384).to(device)

# --- Model Creation (timm + MLPClassifier head) ---
# Create ResNet with num_classes=0 to get pooling output, but we attach the classifier manually 
# to mimic "resnet.fc = MLPClassifier". However, for BYOL_Custom, we wrap it.
# The Wrapper will just use features. The attached head is only if we needed classification output (not used in SSL loss).
# But to match temp2 structure:
from torchvision.models import resnet101, ResNet101_Weights
base_resnet = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)

base_resnet.fc = MLPClassifier(in_features=2048, num_classes=27)

# Now wrap it. Wrapper extracts features from pool, IGNORING the new fc head for SSL purposes.
# This matches BYOL(hidden_layer='avgpool') behavior.
wrapped_model = ResNetWrapper(base_resnet)

# Initialize Learner
learner = BYOL_Custom(wrapped_model, augment_fn=gpu_augment)

# Distributed
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    learner = nn.DataParallel(learner)
learner = learner.to(device)

mim_loss_fn = MIMLoss(min_mask=0.3, max_mask=0.6).to(device)
contrastive_loss_fn = ContrastiveLoss(temperature=0.1).to(device)

min_lr = 1e-6
max_lr = 1e-4
optimizer = optim.AdamW(learner.parameters(), lr=max_lr, weight_decay=1e-5)
scaler = amp.GradScaler('cuda')

def warmup_lr_scheduler(optimizer, warmup_epochs=10, min_lr=1e-6, max_lr=1e-4):
    start_factor = min_lr / max_lr
    def lr_lambda(epoch):
      progress = epoch / warmup_epochs
      factor = start_factor + (1.0 - start_factor) * min(progress, 1.0)
      return factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

warmup_epochs = 10
warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_epochs=warmup_epochs, min_lr=min_lr, max_lr=max_lr)

# --- 6. Training Loop ---
epochs = 90
best_loss = float('inf')

print("Starting Self-Supervised Pretraining...")

for epoch in range(epochs):
    learner.train()
    
    total_loss = 0
    total_byol = 0
    total_mim = 0
    total_contrast = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
    
    for images in pbar:
        images = images.to(device)
        optimizer.zero_grad()
        
        with amp.autocast('cuda', dtype=torch.float32):
            
            # 1. BYOL Loss (Internal Augmentation)
            byol_loss = learner(images) 
            # Handle DataParallel return (vector result)
            if isinstance(byol_loss, torch.Tensor) and byol_loss.ndim > 0:
                byol_loss = byol_loss.mean()

            # 2. MIM Loss
            mim_loss = mim_loss_fn(images, images)

            # 3. Contrastive Loss
            # Augment externally for contrastive views
            aug1 = gpu_augment(images)
            aug2 = gpu_augment(images)
            
            # Get projections (DataParallel friendly)
            z_i = learner(aug1, return_projection=True)
            z_j = learner(aug2, return_projection=True)
            
            contrast_loss = contrastive_loss_fn(z_i, z_j)
            
            # Weighted Sum
            loss = 1.0 * byol_loss + 0.6 * mim_loss + 0.6 * contrast_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Explicit Target Update for DataParallel compatibility
        if isinstance(learner, nn.DataParallel): learner.module.update_target()
        else: learner.update_target()

        total_loss += loss.item()
        total_byol += byol_loss.item()
        total_mim += mim_loss.item()
        total_contrast += contrast_loss.item()

        pbar.set_postfix({
            'Loss': f"{loss.item():.4f}", 
            'BYOL': f"{byol_loss.item():.4f}",
            'MIM': f"{mim_loss.item():.4f}",
            'Cont': f"{contrast_loss.item():.4f}"
        })

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    if epoch < warmup_epochs:
        warmup_scheduler.step()
        print(f"LR Updated: {optimizer.param_groups[0]['lr']:.6f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        model_to_save = learner.module.online_encoder.model if isinstance(learner, nn.DataParallel) else learner.online_encoder.model
        torch.save(model_to_save.state_dict(), "resnet_ssl_best.pth")
        print("Saved Best Model!")

    if (epoch + 1) % 5 == 0:
        model_to_save = learner.module.online_encoder.model if isinstance(learner, nn.DataParallel) else learner.online_encoder.model
        torch.save(model_to_save.state_dict(), f"resnet_ssl_ep{epoch+1}.pth")

print("Pretraining Complete!")