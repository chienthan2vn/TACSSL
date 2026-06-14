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

# -----------------------------------------------------------------------------
# 1. Configuration - EDIT HERE
# -----------------------------------------------------------------------------
@dataclass
class Config:
    # --- Model Settings ---
    base_model_name: str = "resnet10t.c3_in1k"
    feature_dim: int = 128
    weights_path: str = "plantseg.pth"
    device: str = "cpu"  # Set to "cuda" if available
    batch_size: int = 32

    # --- Data Paths ---
    # Directories containing the source images
    path_plantdoc: str = "E:/project/kltn/dataset/src_data/PlantDoc-Dataset/train"
    path_plantwild: str = "E:/project/kltn/dataset/src_data/plantwild/PlantWild/train"
    path_plantseg: str = "E:/project/kltn/dataset/src_data/Plantseg/Plantseg/train"
    path_fieldplant: str = "E:/project/kltn/dataset/src_data/FieldPlant/FieldPlant/train"
    
    # Path used for generating prototypes
    base_path: str = path_plantseg
    
    # --- Output Settings ---
    output_base_dir: str = "plantseg_ssl_dataset_custom_mean_v2"
    embedding_cache_dir: str = "embedding/plantseg_embedding"
    
    # --- Selection Thresholds ---
    # Dictionary mapping dataset name to similarity threshold
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "small": 0.75,
        "medium": 0.72,
        "large": 0.67
    })
    
    # --- Margin Filtering ---
    # Minimum difference between Top-1 and Top-2 similarity scores
    epsilon: float = 0.1

CONFIG = Config()

# Constants
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Transform for embedding extraction
EVAL_TRANSFORM = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# -----------------------------------------------------------------------------
# 2. Model
# -----------------------------------------------------------------------------
class FeatureExtractor(nn.Module):
    """
    Simplified model wrapper for extracting embeddings.
    """
    def __init__(self, base_model_name: str, feature_dim: int):
        super().__init__()
        # Encoder
        self.backbone = timm.create_model(base_model_name, pretrained=True, num_classes=0)
        
        # Detect backbone output dimension
        try:
            enc_dim = self.backbone.conv_head.out_channels
        except AttributeError:
            enc_dim = self.backbone.num_features

        # Projection head
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
        except Exception as e:
            print(f"Warning: Failed to load weights ({e}). Using pretrained backbone.")
    else:
        print(f"Weights file not found at {cfg.weights_path}. Using pretrained backbone.")
    
    model.to(cfg.device)
    model.eval()
    return model

# -----------------------------------------------------------------------------
# 3. Data Loading & Processing
# -----------------------------------------------------------------------------
class ImageListDataset(torch.utils.data.Dataset):
    def __init__(self, paths: List[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), path
        except Exception as e:
            print(f"Error reading {path}: {e}")
            raise e

def find_images_recursive(root_dir: str) -> List[str]:
    """Return sorted list of all image files in directory tree."""
    if not os.path.exists(root_dir):
        return []
    paths = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXTS:
                paths.append(os.path.join(root, f))
    return sorted(paths)

@torch.no_grad()
def compute_embeddings(model: FeatureExtractor, paths: List[str], cfg: Config) -> Tuple[np.ndarray, List[str]]:
    """Compute normalized embeddings for a list of images."""
    if not paths:
        return np.zeros((0, cfg.feature_dim), dtype=np.float32), []
    
    dataset = ImageListDataset(paths, EVAL_TRANSFORM)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    
    embs_list, out_paths = [], []
    for imgs, batch_paths in tqdm(loader, desc="Computing embeddings", leave=False):
        imgs = imgs.to(cfg.device)
        embs = model(imgs).cpu().numpy()
        embs_list.append(embs)
        out_paths.extend(batch_paths)
        
    if not embs_list:
        return np.zeros((0, cfg.feature_dim), dtype=np.float32), []

    embs = np.vstack(embs_list)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / (norms + 1e-9)
    return embs.astype(np.float32), out_paths

# -----------------------------------------------------------------------------
# 4. Deduplication
# -----------------------------------------------------------------------------
def compute_file_hash(path: str, block_size: int = 65536) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read(block_size)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(block_size)
    return hasher.hexdigest()

def remove_duplicates(
    embs: np.ndarray, 
    paths: List[str], 
    cache_dir: str
) -> Tuple[np.ndarray, List[str]]:
    """
    Remove identical images based on MD5 hash.
    Uses a persistent JSON cache to avoid recomputing hashes.
    """
    print("Running deduplication...")
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_path_json = os.path.join(cache_dir, "file_hashes.json")
    
    # Load Cache
    hash_cache = {}
    if os.path.exists(cache_path_json):
        try:
            with open(cache_path_json, "r") as f:
                hash_cache = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load hash cache ({e})")
            
    updates = 0
    unique_indices = []
    seen_hashes = set()
    
    # Process
    for idx, path in enumerate(tqdm(paths, desc="Checking Hashes")):
        try:
            mtime = os.path.getmtime(path)
            # Cache Key: Path
            # Value: {hash: "...", mtime: 12345}
            
            cached_entry = hash_cache.get(path)
            if cached_entry and cached_entry.get("mtime") == mtime:
                file_hash = cached_entry["hash"]
            else:
                # Compute fresh hash
                file_hash = compute_file_hash(path)
                hash_cache[path] = {"hash": file_hash, "mtime": mtime}
                updates += 1
            
            if file_hash not in seen_hashes:
                seen_hashes.add(file_hash)
                unique_indices.append(idx)
        except Exception as e:
            print(f"Error hashing {path}: {e}")
            # Identify as error but dont crash?, skip
            pass

    # Save Cache if changed
    if updates > 0:
        print(f"Updating hash cache with {updates} new entries...")
        with open(cache_path_json, "w") as f:
            json.dump(hash_cache, f)

    # Filter
    unique_indices = np.array(unique_indices)
    print(f"Deduplication: Reduced {len(paths)} -> {len(unique_indices)} images ({len(paths) - len(unique_indices)} duplicates found)")
    
    return embs[unique_indices], [paths[i] for i in unique_indices]


# -----------------------------------------------------------------------------
# 5. Multi-Prototype & Selection Logic (Advanced Margin Filtering)
# -----------------------------------------------------------------------------
def compute_similarity_stats(
    candidate_embs: np.ndarray, 
    candidate_paths: List[str], 
    prototypes_dict: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute similarity statistics for each candidate against its class prototypes.
    """
    num_candidates = len(candidate_paths)
    max_sims = np.zeros(num_candidates, dtype=np.float32)
    margins = np.zeros(num_candidates, dtype=np.float32)

    # Group candidates by class (folder name)
    candidates_by_class = {}
    for idx, path in enumerate(candidate_paths):
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in candidates_by_class:
            candidates_by_class[class_name] = []
        candidates_by_class[class_name].append(idx)

    for class_name, indices in tqdm(candidates_by_class.items(), desc="Calculating Margins"):
        indices = np.array(indices)
        if class_name not in prototypes_dict:
            continue
            
        protos = prototypes_dict[class_name] # (K, dim)
        embs_subset = candidate_embs[indices] # (M, dim)
        
        # Sim Matrix: (M, K)
        sim_matrix = np.dot(embs_subset, protos.T)
        
        K = protos.shape[0]
        if K == 1:
            class_max_sims = sim_matrix.flatten()
            class_margins = np.full_like(class_max_sims, np.inf)
        else:
            if K >= 2:
                partitioned = np.partition(sim_matrix, K-2, axis=1)
                top2_values = partitioned[:, -2:] 
                top2_values.sort(axis=1) # Ascending: [Top2, Top1]
                
                class_max_sims = top2_values[:, 1]
                class_second_sims = top2_values[:, 0]
                class_margins = class_max_sims - class_second_sims
            else:
                pass

        max_sims[indices] = class_max_sims
        margins[indices] = class_margins
        
    return max_sims, margins

def balanced_selection(
    candidate_paths: List[str], 
    max_sims: np.ndarray, 
    margins: np.ndarray,
    threshold: float,
    epsilon: float
) -> Tuple[List[int], Dict]:
    """
    Select images with balancing logic AND margin filtering.
    """
    
    # 1. Filter
    mask = (max_sims >= threshold) & (margins > epsilon)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        return [], {}

    # Group valid indices by class
    class_groups = {}
    for idx in valid_indices:
        path = candidate_paths[idx]
        class_name = os.path.basename(os.path.dirname(path))
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(idx)
        
    # 2. Quota
    total_valid = len(valid_indices)
    num_classes = len(class_groups)
    if num_classes == 0: return [], {}
    
    quota = int(total_valid / num_classes)
    
    selected_indices = []
    remainder_pool = []
    
    # 3. Fill Quota
    for c_name, indices in class_groups.items():
        indices = np.array(indices)
        sims = max_sims[indices]
        sorted_order = np.argsort(-sims)
        sorted_indices = indices[sorted_order]
        
        if len(sorted_indices) <= quota:
            selected_indices.extend(sorted_indices.tolist())
        else:
            selected_indices.extend(sorted_indices[:quota].tolist())
            remainder_pool.extend(sorted_indices[quota:].tolist())
            
    # 4. Backfill
    current_count = len(selected_indices)
    needed = total_valid - current_count
    
    if needed > 0 and len(remainder_pool) > 0:
        remainder_pool = np.array(remainder_pool)
        pool_sims = max_sims[remainder_pool]
        pool_order = np.argsort(-pool_sims)
        
        best_remainders = remainder_pool[pool_order[:needed]].tolist()
        selected_indices.extend(best_remainders)
        
    return selected_indices, {"quota": quota, "total_valid": total_valid}

# -----------------------------------------------------------------------------
# 6. Dataset Saving
# -----------------------------------------------------------------------------
def save_subset(subset_name: str, indices: List[int], all_paths: List[str], all_sims: np.ndarray, cfg: Config):
    """Save selected images and metadata to disk."""
    out_dir = os.path.join(cfg.output_base_dir, subset_name)
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    selected_paths = [all_paths[i] for i in indices]
    selected_sims = all_sims[indices]
    
    # Check stats
    if len(selected_sims) > 0:
        stats = {
            "mean": float(np.mean(selected_sims)),
            "max": float(np.max(selected_sims)),
            "min": float(np.min(selected_sims))
        }
    else:
        stats = {"mean": 0.0, "max": 0.0, "min": 0.0}
        
    print(f"[{subset_name}] Selected {len(selected_paths)}. Stats: {stats}")

    # 1. Manifest
    with open(os.path.join(out_dir, "selected_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(p.replace("\\", "/") for p in selected_paths))
        
    # 2. Copy
    copied, failed = 0, 0
    for i, src in enumerate(tqdm(selected_paths, desc=f"Saving {subset_name}")):
        ext = os.path.splitext(src)[1]
        dst_name = f"image_{i+1}{ext}"
        dst_path = os.path.join(data_dir, dst_name)
        
        suffix = 1
        while os.path.exists(dst_path):
            dst_path = os.path.join(data_dir, f"image_{i+1}_{suffix}{ext}")
            suffix += 1
            
        try:
            shutil.copy2(src, dst_path)
            copied += 1
        except Exception:
            failed += 1
            
    # 3. Meta
    with open(os.path.join(out_dir, "meta.txt"), "w", encoding="utf-8") as f:
        f.write(f"num_selected={len(selected_paths)}\n")
        f.write(f"mean_sim={stats['mean']:.6f}\n")
        f.write(f"max_sim={stats['max']:.6f}\n")
        f.write(f"min_sim={stats['min']:.6f}\n")

# -----------------------------------------------------------------------------
# 7. Main
# -----------------------------------------------------------------------------
def main():
    from make_prototype import get_all_prototypes
    
    cfg = CONFIG
    print(f"--- Dataset Creation Tool (Balanced + Dedupe + Margin) ---\nConfig: {cfg}\n")
    
    # 1. Get Prototypes
    print("\n[Step 1] Loading/Computing Prototypes...")
    prototypes_dict, class_names = get_all_prototypes(cfg)
    print(f"Loaded prototypes for {len(class_names)} classes.")
    
    # 2. Gather Candidates
    print("\n[Step 2] Collecting candidate images...")
    candidate_paths = []
    for path in [cfg.path_plantwild, cfg.path_plantseg, cfg.path_fieldplant, cfg.path_plantdoc]:
        imgs = find_images_recursive(path)
        print(f"  - {path}: {len(imgs)} images")
        candidate_paths.extend(imgs)
        
    if not candidate_paths: return

    # 3. Compute/Load Embeddings
    print("\n[Step 3] Computing embeddings for candidates...")
    model = load_model(cfg)
    os.makedirs(cfg.embedding_cache_dir, exist_ok=True)
    cache_file = os.path.join(cfg.embedding_cache_dir, "embeddings_candidates.npz")
    
    if os.path.exists(cache_file):
        print(" -> Loading candidates from cache")
        data = np.load(cache_file, allow_pickle=True)
        embs = data["embs"]
        # In real scenario, verify paths match
    else:
        print(" -> Computing fresh embeddings")
        embs, _ = compute_embeddings(model, candidate_paths, cfg)
        np.savez_compressed(cache_file, embs=embs, paths=candidate_paths)

    # 4. Remove Duplicates (NEW STEP)
    # We filter both 'embs' and 'candidate_paths'
    print("\n[Step 4] Removing Duplicates...")
    embs, candidate_paths = remove_duplicates(embs, candidate_paths, cfg.embedding_cache_dir)

    # 5. Compute Similarity Stats
    print("\n[Step 5] Computing Similarities & Margins...")
    max_sims, margins = compute_similarity_stats(embs, candidate_paths, prototypes_dict)
    
    # 6. Selection
    print("\n[Step 6] Balanced Selection (Margin > {:.2f})...".format(cfg.epsilon))
    for name, threshold in cfg.thresholds.items():
        print(f"\nSubset '{name}' (Threshold >= {threshold})")
        
        indices, debug_info = balanced_selection(candidate_paths, max_sims, margins, threshold, cfg.epsilon)
        
        if indices:
            print(f" -> Quota per class was approx: {debug_info.get('quota')}")
            save_subset(name, indices, candidate_paths, max_sims, cfg)
        else:
            print(" -> No images selected.")

if __name__ == "__main__":
    main()
