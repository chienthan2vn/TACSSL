import os
import shutil
import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm

from sklearn.neighbors import NearestNeighbors

# -----------------------------------------------------------------------------
# 1. Configuration - EDIT HERE
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # --- Model Settings ---
    base_model_name: str = "resnet10t.c3_in1k"
    feature_dim: int = 128
    weights_path: str = "plantdoc.pth"
    device: str = "cpu"  # Set to "cuda" if available
    batch_size: int = 32

    # --- Data Paths ---
    ssl_data_path: str = "E:/project/kltn/dataset/ssl_data_filter"
    path_plantdoc: str = "E:/project/kltn/dataset/src_data/PlantDoc-Dataset/train"
    path_plantwild: str = "E:/project/kltn/dataset/src_data/plantwild/PlantWild/train"
    path_plantseg: str = "E:/project/kltn/dataset/src_data/Plantseg/Plantseg/train"
    path_fieldplant: str = "E:/project/kltn/dataset/src_data/FieldPlant/FieldPlant/train"
    
    base_path: str = path_plantdoc # Deprecated
    
    # --- Output Settings ---
    output_base_dir: str = "plantdoc_ssl_dataset_knn_v1"
    embedding_cache_dir: str = "embedding/plantdoc_embedding_knn"
    
    # --- Selection Thresholds (VOTING RATIO) ---
    # 0.5 means 50% of the k nearest neighbors must belong to the same class.
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "small": 0.48,
        "medium": 0.45,
        "large": 0.4
    })
    
    # --- Adaptive Filtering ---
    # percentile: float = 70.0
    
    # Sigma for Gaussian Kernel (Distance Weighting)
    # sigma = 0.5: Sharp drop-off (strict). sigma = 1.0: Smoother.
    sigma: float = 0.5
    
    # Minimum Density Threshold (Out-of-Distribution Rejection)
    # If density_sum < threshold, the image is too far from ALL prototypes.
    min_density_threshold: float = 0.39

CONFIG = Config()

# Constants
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Transform
EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# 2. Model (Same as main.py)
# -----------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, base_model_name: str, feature_dim: int):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        try:
            enc_dim = self.backbone.conv_head.out_channels
        except AttributeError:
            enc_dim = self.backbone.num_features

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
        try:
            state_dict = torch.load(cfg.weights_path, map_location=cfg.device)
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded weights from {cfg.weights_path}")
        except Exception:
            print(f"Warning: Failed to load weights. Using pretrained backbone.")
    else:
        print(f"Weights file not found at {cfg.weights_path}. Using pretrained backbone.")
    model.to(cfg.device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 3. Data Loading
# -----------------------------------------------------------------------------
class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
            return self.transform(img), self.paths[idx]
        except Exception as e:
            print(f"Error {self.paths[idx]}: {e}")
            raise e

def find_images_recursive(root_dir: str) -> List[str]:
    if not os.path.exists(root_dir): return []
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)

@torch.no_grad()
def compute_embeddings(model: FeatureExtractor, paths: List[str], cfg: Config) -> Tuple[np.ndarray, List[str]]:
    if not paths: return np.zeros((0, cfg.feature_dim), dtype=np.float32), []
    dataset = ImageListDataset(paths, EVAL_TRANSFORM)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    embs_list, out_paths = [], []
    for imgs, batch_paths in tqdm(loader, desc="Computing embeddings", leave=False):
        imgs = imgs.to(cfg.device)
        embs = model(imgs).cpu().numpy()
        embs_list.append(embs)
        out_paths.extend(batch_paths)
    if not embs_list: return np.zeros((0, cfg.feature_dim), dtype=np.float32), []
    embs = np.vstack(embs_list)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return (embs / (norms + 1e-9)).astype(np.float32), out_paths

# -----------------------------------------------------------------------------
# 4. Deduplication
# -----------------------------------------------------------------------------
def compute_file_hash(path: str, block_size: int = 65536) -> str:
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def remove_duplicates(embs: np.ndarray, paths: List[str], cache_dir: str) -> Tuple[np.ndarray, List[str]]:
    print("Running deduplication...")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path_json = os.path.join(cache_dir, "file_hashes.json")
    hash_cache = {}
    if os.path.exists(cache_path_json):
        try:
            with open(cache_path_json, "r") as f: hash_cache = json.load(f)
        except: pass
            
    updates = 0
    unique_indices = []
    seen_hashes = set()
    
    for idx, path in enumerate(tqdm(paths, desc="Checking Hashes")):
        try:
            mtime = os.path.getmtime(path)
            cached_entry = hash_cache.get(path)
            if cached_entry and cached_entry.get("mtime") == mtime:
                file_hash = cached_entry["hash"]
            else:
                file_hash = compute_file_hash(path)
                hash_cache[path] = {"hash": file_hash, "mtime": mtime}
                updates += 1
            
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_indices.append(idx)
        except: pass

    if updates > 0:
        with open(cache_path_json, "w") as f: json.dump(hash_cache, f)

    print(f"Deduplication: Reduced {len(paths)} -> {len(unique_indices)}")
    return embs[unique_indices], [paths[i] for i in unique_indices]

# -----------------------------------------------------------------------------
# 5. k-NN VOTING LOGIC (New)
# -----------------------------------------------------------------------------
def compute_knn_voting_scores(
    candidate_embs: np.ndarray, 
    candidate_paths: List[str], 
    prototypes_dict: Dict[str, np.ndarray],
    # percentile: float = 70.0
) -> np.ndarray:
    """
    Compute Class-Normalized Density scores for each candidate.
    Score = Density_target / Sum(Density_all_classes_in_neighborhood)
    Density_c = (Sum of weights of neighbors belonging to class c) / (Total prototypes of class c)
    """
    
    # 1. Flatten Prototypes & Pre-compute Class Counts
    global_protos_list = []
    global_labels_list = []
    class_counts = {}
    
    print("Building Global Prototype Index...")
    for class_name, protos in prototypes_dict.items():
        count = protos.shape[0]
        class_counts[class_name] = count
        # protos shape (K, dim)
        for i in range(count):
            global_protos_list.append(protos[i])
            global_labels_list.append(class_name)
            
    global_protos = np.array(global_protos_list) # (Total_P, dim)
    global_labels = np.array(global_labels_list) # (Total_P,)
    
    Total_N = global_protos.shape[0]
    print(f"Total Prototypes: {Total_N}")
    
    # 2. Determine k for k-NN
    # k = sqrt(N)
    k_neighbors = int(np.sqrt(Total_N))
    if k_neighbors % 2 == 0:
        k_neighbors += 1 # Ensure odd
    print(f"Using k={k_neighbors} for voting.")
    
    # 3. Fit Nearest Neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors, algorithm='brute', metric='cosine').fit(global_protos)
    
    # 4. Query
    print("Querying k-NN with Class-Normalized Density Scoring...")
    
    batch_size = 1000
    num_candidates = len(candidate_paths)
    voting_scores = np.zeros(num_candidates, dtype=np.float32)
    # epsilon = 1e-6 # No longer needed for Gaussian
    sigma = getattr(CONFIG, 'sigma', 0.5) 
    sigma_sq_2 = 2 * (sigma ** 2)
    min_density_thresh = getattr(CONFIG, 'min_density_threshold', 0.001)
    
    for start_idx in tqdm(range(0, num_candidates, batch_size), desc="Voting"):
        end_idx = min(start_idx + batch_size, num_candidates)
        batch_embs = candidate_embs[start_idx:end_idx]
        batch_paths = candidate_paths[start_idx:end_idx]
        
        # Returns distances and indices
        dists, indices = nbrs.kneighbors(batch_embs)
        
        # --- Adaptive Percentile Cutoff ---
        # Calculate cutoff for each candidate in the batch
        # dists shape: (batch, k)
        # cutoff_vals = np.percentile(dists, percentile, axis=1, keepdims=True)
        
        # Create mask: keep neighbors strictly within percentile
        # mask_cutoff = (dists <= cutoff_vals).astype(np.float32)

        # Compute weights: Gaussian Kernel
        # w = exp(- d^2 / 2*sigma^2)
        # dists is cosine distance (0..2).
        weights = np.exp(- (dists ** 2) / sigma_sq_2)
        
        # Apply cutoff mask
        # weights = weights * mask_cutoff
        
        # Resolve labels
        neighbor_classes = global_labels[indices] # (batch, k)
        
        # Calculate scores - UNLABELED SELECTION LOGIC
        # We assume the image belongs to the class with the highest density in the neighborhood.
        for i in range(len(batch_paths)):
            abs_idx = start_idx + i
            # Neighbors for this image
            my_neighbor_classes = neighbor_classes[i] # (k,)
            my_weights = weights[i] # (k,)
            
            path = batch_paths[i]

            # Identify unique classes in this neighborhood
            unique_classes = np.unique(my_neighbor_classes)
            
            densities = []
            
            for cls_name in unique_classes:
                # Sum weights for this specific class
                cls_mask = (my_neighbor_classes == cls_name)
                sum_w_cls = np.sum(my_weights * cls_mask)
                
                # Normalize by total prototypes of this class (N_c)
                count_c = class_counts.get(cls_name, 1)
                density_cls = sum_w_cls / count_c
                
                densities.append(density_cls)
            
            if not densities:
                voting_scores[abs_idx] = 0.0
                continue

            # NEW LOGIC: Max Probability
            # Score = Max(Density_c) / Sum(Density_all)
            density_max = max(densities)
            density_sum = sum(densities)
            
            # Check Out-of-Distribution
            if density_max < min_density_thresh:
                voting_scores[abs_idx] = 0.0
            else:
                voting_scores[abs_idx] = density_max / density_sum
            
    return voting_scores

def confidence_selection(
    candidate_paths: List[str], 
    voting_scores: np.ndarray, 
    threshold: float
) -> Tuple[List[int], Dict]:
    """
    Select images strictly based on Voting Confidence Score.
    No quota balancing.
    """
    # Filter
    mask = voting_scores > threshold
    selected_indices = np.where(mask)[0].tolist()
    
    # Compute minimal stats
    if not selected_indices:
        return [], {}
        
    return selected_indices, {}

# -----------------------------------------------------------------------------
# 6. Saving
# -----------------------------------------------------------------------------
def save_subset(subset_name: str, indices: List[int], all_paths: List[str], all_scores: np.ndarray, cfg: Config):
    out_dir = os.path.join(cfg.output_base_dir, subset_name)
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    selected_paths = [all_paths[i] for i in indices]
    selected_scores = all_scores[indices]
    
    stats = {
        "mean_score": float(np.mean(selected_scores)),
        "min_score": float(np.min(selected_scores)),
        "max_score": float(np.max(selected_scores)),
        "count": len(selected_paths)
    }
    print(f"[{subset_name}] Selected {len(selected_paths)}. Stats: {stats}")

    with open(os.path.join(out_dir, "selected_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(p.replace("\\", "/") for p in selected_paths))
        
    copied = 0
    for i, src in enumerate(tqdm(selected_paths, desc=f"Saving {subset_name}")):
        ext = os.path.splitext(src)[1]
        dst_path = os.path.join(data_dir, f"image_{i+1}{ext}")
        suffix = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(data_dir, f"image_{i+1}_{suffix}{ext}")
            suffix += 1
        try:
            shutil.copy2(src, dst_path)
            copied += 1
        except: pass
            
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"num_selected={len(selected_paths)}\n")
        f.write(f"mean_voting_score={stats['mean_score']:.4f}\n")
        f.write(f"min_voting_score={stats['min_score']:.4f}\n")
        f.write(f"max_voting_score={stats['max_score']:.4f}\n")

# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------
def main():
    from make_prototype import get_all_prototypes
    
    cfg = CONFIG
    print(f"--- Dataset Creation Tool V2 (Density-Based k-NN Voting) ---\nConfig: {cfg}\n")
    
    # 1. Get Prototypes
    print("\n[Step 1] Loading/Computing Prototypes...")
    prototypes_dict, class_names = get_all_prototypes(cfg)
    
    # 2. Gather Candidates
    print("\n[Step 2] Collecting candidate images...")
    candidate_paths = []
    imgs = find_images_recursive(cfg.ssl_data_path)
    print(f"  - {cfg.ssl_data_path}: {len(imgs)} images")
    candidate_paths.extend(imgs)
    if not candidate_paths: return

    # 3. Compute Embeddings
    print("\n[Step 3] Computing embeddings...")
    model = load_model(cfg)
    os.makedirs(cfg.embedding_cache_dir, exist_ok=True)
    cache_file = os.path.join(cfg.embedding_cache_dir, "embeddings_candidates.npz")
    
    if os.path.exists(cache_file):
        print(" -> Loading from cache")
        data = np.load(cache_file, allow_pickle=True)
        embs = data["embs"]
    else:
        print(" -> Computing fresh")
        embs, _ = compute_embeddings(model, candidate_paths, cfg)
        np.savez_compressed(cache_file, embs=embs, paths=candidate_paths)

    # 4. Deduplication
    embs, candidate_paths = remove_duplicates(embs, candidate_paths, cfg.embedding_cache_dir)

    # 5. k-NN Voting (New Logic)
    print("\n[Step 5] Computing k-NN Voting Scores...")
    voting_scores = compute_knn_voting_scores(embs, candidate_paths, prototypes_dict)
    
    # 6. Strict Selection
    print("\n[Step 6] Density-Based Selection...")
    for name, threshold in cfg.thresholds.items():
        print(f"\nSubset '{name}' (Voting Score > {threshold})")
        
        indices, _ = confidence_selection(candidate_paths, voting_scores, threshold)
        
        if indices:
            save_subset(name, indices, candidate_paths, voting_scores, cfg)
        else:
            print(" -> No images selected.")

if __name__ == "__main__":
    main()
