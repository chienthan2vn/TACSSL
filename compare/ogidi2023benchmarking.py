import os
import shutil
import json
import math
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
from sklearn.cluster import MiniBatchKMeans, KMeans

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # --- Paths ---
    ssl_data_path: str = "E:/project/kltn/dataset/ssl_data_filter"
    weights_path: str = "E:/project/kltn/dataset/plantdoc.pth"
    output_dir: str = "E:/project/kltn/dataset/facebook/curation_output"
    cache_dir: str = "E:/project/kltn/dataset/facebook/cache"
    
    # --- Model Params ---
    base_model_name: str = "resnet10t.c3_in1k"
    feature_dim: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 128
    
    # --- Curation / Pruning Params ---
    target_ratio: float = 0.07
    
    # Hierarchical K-Means Settings (Bottom-Up)
    # L1: Fine clustering (Data -> Clusters). High K.
    # L2: Coarse clustering (L1 Centroids -> Super Clusters). Low K.
    # Example: 20k images.
    # L1: 20k / 50 = 400 clusters.
    # L2: 400 / 10 = 40 clusters.
    cluster_ratio_l1: int = 50   # Avg images per bucket in L1
    cluster_ratio_l2: int = 10   # Avg L1 buckets per L2 bucket
    
    # Resampling Loop
    resampling_iters: int = 10 # Paper uses higher iters for stability
    resampling_sample_fraction: float = 0.5 
    
    # Random Seed
    seed: int = 42

CONFIG = Config()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# 2. MODEL DEFINITION
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
    print(f"[Model] Loading '{cfg.base_model_name}'...")
    model = FeatureExtractor(cfg.base_model_name, cfg.feature_dim)
    if os.path.exists(cfg.weights_path):
        try:
            state_dict = torch.load(cfg.weights_path, map_location=cfg.device)
            model.load_state_dict(state_dict, strict=False)
            print(f"  [+] Loaded weights from {cfg.weights_path}")
        except Exception as e:
            print(f"  [!] Failed to load weights ({e}).")
    else:
        print(f"  [!] Weights file not found at {cfg.weights_path}.")
    model.to(cfg.device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 3. DATA & EMBEDDINGS
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
        except:
            return torch.zeros((3, 224, 224)), ""

def find_images_recursive(root_dir: str) -> List[str]:
    if not os.path.exists(root_dir): return []
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)

@torch.no_grad()
def get_embeddings(paths: List[str], cfg: Config) -> np.ndarray:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    cache_path = os.path.join(cfg.cache_dir, f"embeddings_{len(paths)}_{cfg.base_model_name}.npz")
    
    if os.path.exists(cache_path):
        print(f"[Embeddings] Loading from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        if len(data["paths"]) == len(paths):
            return data["embs"]
            
    print("[Embeddings] Computing fresh...")
    model = load_model(cfg)
    dataset = ImageListDataset(paths, EVAL_TRANSFORM)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=2, shuffle=False)
    
    embs_list = []
    for imgs, batch_paths in tqdm(loader):
        imgs = imgs.to(cfg.device)
        if batch_paths[0] == "": continue 
        feats = model(imgs).cpu().numpy()
        embs_list.append(feats)
    
    embeddings = np.vstack(embs_list)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)
    
    np.savez_compressed(cache_path, embs=embeddings, paths=paths)
    return embeddings

# -----------------------------------------------------------------------------
# 4. HIERARCHICAL K-MEANS WITH RESAMPLING (ALGORITHM 1)
# -----------------------------------------------------------------------------
def run_kmeans_resampling(data: np.ndarray, k: int, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the inner loop of Algorithm 1:
    - Input: Data I (or Centroids), K
    - Returns: Labels, Centroids
    """
    print(f"  -> Clustering {len(data)} items into {k} clusters (Resampling={cfg.resampling_iters})...")
    
    # Init
    # Use MiniBatchKMeans for speed, it approximates KMeans
    kmeans = MiniBatchKMeans(
        n_clusters=k, 
        init='k-means++', 
        batch_size=min(2048, len(data)),
        random_state=cfg.seed,
        n_init=3
    )
    labels = kmeans.fit_predict(data)
    centroids = kmeans.cluster_centers_
    
    # Resampling Loop (Algorithm 1 lines 6-10)
    if cfg.resampling_iters > 0:
        for i in range(cfg.resampling_iters):
            # 1. Resample (line 7): sample r_t points from each cluster
            # We construct R = Union of closest points
            
            # Distance to assigned centroid
            # assigned_centroids = centroids[labels]
            # dists = ((data - assigned_centroids)**2).sum(axis=1) # Squared Euclidean
            
            # Efficient implementation: iterate clusters
            indices_to_keep = []
            
            # Pre-compute distances for speed? 
            # Or just use the fact that we can iterate.
            # To strictly follow "sample points from each cluster", we find the members.
            
            # We need to know distance of members to their centroid
            # Let's do it per cluster to save memory
            for c_idx in range(k):
                member_idxs = np.where(labels == c_idx)[0]
                if len(member_idxs) == 0: continue
                
                # Get data for this cluster
                cluster_data = data[member_idxs]
                centroid = centroids[c_idx]
                
                # Distances
                dists = ((cluster_data - centroid)**2).sum(axis=1)
                
                # Select top fraction
                n_keep = max(1, int(len(member_idxs) * cfg.resampling_sample_fraction))
                
                # Get smallest dist indices
                # partition is faster than argsort for top-k
                if n_keep < len(dists):
                    kept_local_args = np.argpartition(dists, n_keep)[:n_keep]
                else:
                    kept_local_args = np.arange(len(dists))
                    
                indices_to_keep.extend(member_idxs[kept_local_args])
            
            R = data[indices_to_keep]
            
            # 2. Update Centroids (line 8): kmeans(R, k)
            # We initialize with old centroids for stability
            if len(R) >= k:
                kmeans = MiniBatchKMeans(
                    n_clusters=k,
                    init=centroids,
                    batch_size=min(2048, len(R)),
                    random_state=cfg.seed,
                    n_init=1
                ).fit(R)
                centroids = kmeans.cluster_centers_
            
            # 3. Re-assign whole dataset (line 9)
            labels = kmeans.predict(data)
            
    return labels, centroids

def hierarchical_clustering(embeddings: np.ndarray, cfg: Config) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Implements 'Algorithm 1' outer loop (Bottom-Up).
    Returns list of results per level: [(labels_1, centroids_1), (labels_2, centroids_2), ...]
    """
    results = []
    current_input = embeddings
    N = len(embeddings)
    
    # Determine K for levels
    # Bottom-up definition:
    # Level 1 (Fine): Data -> C1. K1 = N / ratio1.
    # Level 2 (Coarse): C1 -> C2. K2 = K1 / ratio2.
    
    k1 = max(2, int(N / cfg.cluster_ratio_l1))
    k2 = max(2, int(k1 / cfg.cluster_ratio_l2))
    
    Ks = [k1, k2]
    
    print("\n[Clustering] Starting Bottom-Up Hierarchical K-Means...")
    
    for t_idx, k_t in enumerate(Ks):
        level = t_idx + 1
        print(f"\n[Level {level}] Input Size: {len(current_input)}. Target Clusters: {k_t}")
        
        # Run clustering
        labels, centroids = run_kmeans_resampling(current_input, k_t, cfg)
        
        # Store
        results.append((labels, centroids))
        
        # Prepare input for next level (Centroids become input)
        current_input = centroids
        
    return results

# -----------------------------------------------------------------------------
# 5. HIERARCHICAL SAMPLING (RECONSTRUCTED)
# -----------------------------------------------------------------------------
def binary_search_capping(cluster_sizes: List[int], target_total: int) -> int:
    if not cluster_sizes: return 0
    low = 0
    high = max(cluster_sizes)
    best_mu = 0
    while low <= high:
        mid = (low + high) // 2
        total = sum(min(s, mid) for s in cluster_sizes)
        if total <= target_total:
            best_mu = mid; low = mid + 1
        else:
            high = mid - 1
    return best_mu

def apply_hierarchical_sampling(
    hierarchy_results: List[Tuple[np.ndarray, np.ndarray]], 
    num_data_points: int, 
    cfg: Config
) -> List[int]:
    """
    Reconstructs the tree structure from bottom-up results and applies top-down sampling.
    Hierarchy: 
      0: (L1_labels, L1_centroids) -> Maps Data to L1
      1: (L2_labels, L2_centroids) -> Maps L1 to L2
    """
    
    l1_labels, _ = hierarchy_results[0] # Length N, values 0..K1-1
    l2_labels, _ = hierarchy_results[1] # Length K1, values 0..K2-1
    
    # Build Tree: L2_ID -> List of L1_IDs -> List of Data_Indices
    tree = {}
    
    # 1. Map L2 -> L1
    l2_to_l1 = {} # {l2_id: [l1_idx, ...]}
    for l1_idx, l2_id in enumerate(l2_labels):
        if l2_id not in l2_to_l1: l2_to_l1[l2_id] = []
        l2_to_l1[l2_id].append(l1_idx)
        
    # 2. Map L1 -> Data
    l1_to_data = {} # {l1_idx: [data_idx, ...]}
    print("\n[Sampling] Mapping Data to Clusters...")
    # Faster than loop:
    for data_idx, l1_id in enumerate(l1_labels):
        if l1_id not in l1_to_data: l1_to_data[l1_id] = []
        l1_to_data[l1_id].append(data_idx)
        
    # 3. Top-Down Budget Allocation
    
    target_total = int(num_data_points * cfg.target_ratio)
    print(f"[Sampling] Target Total: {target_total}")
    
    selected_indices = []
    
    # --- Level 2 (Coarsest) ---
    # Concept: We want to cap the number of *Images* contributing to each L2 cluster.
    # Calculate total images in each L2 cluster
    l2_counts = []
    sorted_l2_ids = sorted(l2_to_l1.keys())
    
    for l2_id in sorted_l2_ids:
        # Sum of images in all child L1s
        count = sum(len(l1_to_data.get(l1_id, [])) for l1_id in l2_to_l1[l2_id])
        l2_counts.append(count)
        
    # Find Mu 2
    mu_2 = binary_search_capping(l2_counts, target_total)
    print(f"  -> Level 2 Cap (Mu_2): {mu_2} images per Super-Cluster")
    
    # --- Level 1 (Finer) ---
    # For each L2 cluster, we have a budget of min(count, mu_2).
    # We distribute this budget to its children L1 clusters using Mu_1.
    
    for i, l2_id in enumerate(sorted_l2_ids):
        l2_total_images = l2_counts[i]
        l2_budget = min(l2_total_images, mu_2)
        
        if l2_budget == 0: continue
        
        # Get children L1 clusters
        child_l1_ids = l2_to_l1[l2_id]
        l1_image_counts = [len(l1_to_data.get(lid, [])) for lid in child_l1_ids]
        
        # Find Mu 1 for this specific L2 group
        # (Local capping within the super-cluster to ensure fairness among sub-concepts)
        mu_1 = binary_search_capping(l1_image_counts, l2_budget)
        
        # Select images
        for j, l1_id in enumerate(child_l1_ids):
            l1_count = l1_image_counts[j]
            n_keep = min(l1_count, mu_1)
            
            if n_keep > 0:
                candidates = l1_to_data.get(l1_id, [])
                # Random sampling ("r")
                rng = np.random.default_rng(cfg.seed + l2_id + l1_id)
                picked = rng.choice(candidates, size=n_keep, replace=False)
                selected_indices.extend(picked)

    return selected_indices

# -----------------------------------------------------------------------------
# 6. MAIN PIPELINE
# -----------------------------------------------------------------------------
def main():
    print("--- Automatic Data Curation Pipeline (Algorithm 1) ---")
    cfg = CONFIG
    print(f"Config: {cfg}\n")
    
    # 1. Discovery
    paths = find_images_recursive(cfg.ssl_data_path)
    print(f"Found {len(paths)} images.")
    if not paths: return
    
    # 2. Embeddings
    embs = get_embeddings(paths, cfg)
    
    # 3. Hierarchical KMeans (Bottom-Up)
    hierarchy = hierarchical_clustering(embs, cfg)
    
    # 4. Hierarchical Sampling
    selected_indices = apply_hierarchical_sampling(hierarchy, len(paths), cfg)
    
    # 5. Save
    print(f"\n[Save] Selected {len(selected_indices)} images.")
    os.makedirs(cfg.output_dir, exist_ok=True)
    
    out_list = [paths[i] for i in selected_indices]
    with open(os.path.join(cfg.output_dir, "curated_list.txt"), "w") as f:
        f.write("\n".join(out_list))
        
    # Copy?
    # for p in out_list: shutil.copy(...)
    
    print("Done.")

if __name__ == "__main__":
    main()
