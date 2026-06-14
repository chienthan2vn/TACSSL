import os
import shutil
import json
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

from sklearn.cluster import KMeans

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # --- Paths ---
    # Path to the pretrained weights (plantdoc.pth)
    weights_path: str = "E:/project/kltn/dataset/plantdoc.pth"
    # Source directory containing images to prune
    source_data_path: str = "E:\project\kltn\dataset\ssl_data_filter"
    # Output directory for pruned dataset
    output_path: str = "E:/project/kltn/dataset/beyond/pruned_dataset"
    
    # --- Strategy Settings ---
    # 'easy': Keep images CLOSEST to centroids (Scarce Data strategy)
    # 'hard': Keep images FARTHEST from centroids (Abundant Data strategy)
    pruning_mode: str = "easy" 
    
    # Fraction of data to KEEP (0.0 to 1.0)
    # e.g., 0.7 means keep 70% of the data PER CLASS
    keep_fraction: float = 0.2
    
    # --- Model Settings ---
    base_model_name: str = "resnet10t.c3_in1k"
    feature_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 32
    num_workers: int = 4
    
    # --- Clustering Settings ---
    # Number of prototypes (clusters)
    num_prototypes: int = 200 

    def __post_init__(self):
        print(f"--- Configuration ---")
        print(f"Device: {self.device}")
        print(f"Mode: {self.pruning_mode}")
        print(f"Keep Fraction: {self.keep_fraction}")
        print(f"Prototypes: {self.num_prototypes}")
        print(f"Weights: {self.weights_path}")
        print(f"Source: {self.source_data_path}")
        print(f"Output: {self.output_path}")
        print(f"---------------------")

# -----------------------------------------------------------------------------
# 2. Model Architecture (Replicated from main_v3.py)
# -----------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, base_model_name: str, feature_dim: int):
        super().__init__()
        # Load backbone with no classifier
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        
        # Determine output dimension of the backbone
        try:
            enc_dim = self.backbone.conv_head.out_channels
        except AttributeError:
            enc_dim = self.backbone.num_features

        # Projection head: Linear -> ReLU -> Linear
        self.ssl_head = nn.Sequential(
            nn.Linear(enc_dim, enc_dim * 2),
            nn.ReLU(),
            nn.Linear(enc_dim * 2, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        proj = self.ssl_head(features)
        return F.normalize(proj, dim=1)

def load_model(cfg: Config) -> FeatureExtractor:
    model = FeatureExtractor(cfg.base_model_name, cfg.feature_dim)
    
    if os.path.exists(cfg.weights_path):
        print(f"Loading weights from {cfg.weights_path}...")
        try:
            state_dict = torch.load(cfg.weights_path, map_location=cfg.device)
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Using pretrained backbone weights instead.")
    else:
        print(f"Weights file not found at {cfg.weights_path}. Using pretrained backbone weights.")
        
    model.to(cfg.device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 3. Data Loading & Processing
# -----------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a dummy tensor slightly hacky but prevents crash in batch
            return torch.zeros((3, 224, 224)), path

def get_transform():
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def find_images(root_dir: str) -> List[str]:
    print(f"Scanning for images in {root_dir}...")
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(root, f))
    paths.sort()
    print(f"Found {len(paths)} images.")
    return paths

@torch.no_grad()
def compute_embeddings(model: FeatureExtractor, paths: List[str], cfg: Config) -> np.ndarray:
    if not paths:
        return np.zeros((0, cfg.feature_dim), dtype=np.float32)
        
    dataset = ImageListDataset(paths, get_transform())
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size, 
        num_workers=cfg.num_workers, 
        shuffle=False
    )
    
    embs_list = []
    print("Computing embeddings...")
    for imgs, _ in tqdm(loader):
        imgs = imgs.to(cfg.device)
        embs = model(imgs).cpu().numpy()
        embs_list.append(embs)
        
    return np.vstack(embs_list)

# -----------------------------------------------------------------------------
# 4. Clustering & Scoring
# -----------------------------------------------------------------------------
def perform_clustering(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    print(f"Clustering into {n_clusters} prototypes using KMeans...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    return kmeans.cluster_centers_

def calculate_scores(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    print("Calculating distances to nearest prototypes...")
    N = embeddings.shape[0]
    min_dists = np.zeros(N, dtype=np.float32)
    
    batch_size = 1000
    for i in range(0, N, batch_size):
        end = min(i + batch_size, N)
        batch = embeddings[i:end] # (B, D)
        dists = np.linalg.norm(batch[:, None, :] - centroids[None, :, :], axis=2)
        min_dists[i:end] = np.min(dists, axis=1)
        
    return min_dists

# -----------------------------------------------------------------------------
# 5. Pruning Logic & Execution
# -----------------------------------------------------------------------------
def extract_classname(path: str) -> str:
    # Assumes structure: .../classname/image.jpg
    return os.path.basename(os.path.dirname(path))

def run_pruning(cfg: Config):
    # 1. Setup
    image_paths = find_images(cfg.source_data_path)
    if not image_paths:
        print("No images found. Exiting.")
        return

    # 2. Model & Embeddings
    model = load_model(cfg)
    embeddings = compute_embeddings(model, image_paths, cfg)
    
    # 3. Clustering
    prototypes = perform_clustering(embeddings, cfg.num_prototypes)
    
    # 4. Scoring (Euclidean Distance to Nearest Centroid)
    scores = calculate_scores(embeddings, prototypes)
    
    # 5. Grouping by Class for Balanced Selection
    print("Grouping data by class...")
    class_groups = {} # {classname: [(index, score)]}
    
    for idx, path in enumerate(image_paths):
        cls_name = extract_classname(path)
        if cls_name not in class_groups:
            class_groups[cls_name] = []
        class_groups[cls_name].append((idx, scores[idx]))
        
    # 6. Per-Class Pruning
    selected_indices = []
    stats_per_class = {}
    
    print(f"Applying pruning (Mode: {cfg.pruning_mode}, Fraction: {cfg.keep_fraction}) per class...")
    
    for cls_name, items in class_groups.items():
        # Sort based on strategy
        if cfg.pruning_mode == "easy":
            # Keep smallest distances
            items.sort(key=lambda x: x[1]) 
        elif cfg.pruning_mode == "hard":
            # Keep largest distances
            items.sort(key=lambda x: x[1], reverse=True)
            
        # Determine strict cut-off
        n_total = len(items)
        n_keep = max(1, int(n_total * cfg.keep_fraction)) # Ensure at least 1 image if possible
        
        # Select
        kept_items = items[:n_keep]
        for idx, _ in kept_items:
            selected_indices.append(idx)
            
        stats_per_class[cls_name] = {
            "original": n_total,
            "kept": len(kept_items)
        }

    selected_indices.sort()
    print(f"Total Selected: {len(selected_indices)} / {len(image_paths)} images.")
    
    # 7. Save/Copy Results
    if os.path.exists(cfg.output_path):
        print(f"Cleaning output directory: {cfg.output_path}")
        shutil.rmtree(cfg.output_path)
    os.makedirs(cfg.output_path, exist_ok=True)
    
    report_data = []
    
    print("Copying files...")
    for idx in tqdm(selected_indices):
        src_path = image_paths[idx]
        score = float(scores[idx])
        cls_name = extract_classname(src_path)
        
        # Create class subdirectory in output (Preserving structure)
        # OR just flat? Usually preserving structure is better for datasets.
        # Let's preserve class structure: output/classname/image.jpg
        dst_dir = os.path.join(cfg.output_path, cls_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        fname = os.path.basename(src_path)
        dst_path = os.path.join(dst_dir, fname)
        
        # Handle duplicates if filenames clash in same class (unlikely but safe)
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(fname)
            dst_path = os.path.join(dst_dir, f"{base}_{idx}{ext}")
            
        shutil.copy2(src_path, dst_path)
        report_data.append({
            "file": fname,
            "class": cls_name,
            "original_path": src_path,
            "score_distance": score
        })
        
    # Save Report
    report_path = os.path.join(cfg.output_path, "pruning_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "config": {
                "mode": cfg.pruning_mode,
                "keep_fraction": cfg.keep_fraction,
                "num_prototypes": cfg.num_prototypes
            },
            "stats": {
                "total_original": len(image_paths),
                "total_selected": len(selected_indices),
                "class_breakdown": stats_per_class
            },
            # "selected_files": report_data # Commented out to save space if dataset is huge
        }, f, indent=2)
        
    print(f"Done! Report saved to {report_path}")

if __name__ == "__main__":
    # Settings are now controlled purely by Config class
    cfg = Config()
    run_pruning(cfg)
