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
import torchvision.models as models
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
    weights_path: str = "model/resnet10t.c3_in1k_alpha_0.5_beta_1.0_best.pth"
    device: str = "cpu"  # Set to "cuda" if available
    batch_size: int = 32

    # --- Data Paths ---
    ssl_data_path: List[str] = field(default_factory=lambda: [
        "dataset/src_data/ssl_data"
        ])
    path_plantdoc: str = "dataset/src_data/plantdoc/train"
    path_plantwild: str = "E:/project/dataset/src_data/plantwild/images"
    path_plantseg: str = "E:/project/dataset/src_data/plantseg/images"
    path_fieldplant: str = "E:/project/dataset/src_data/FieldPlant/FieldPlant/train"
    
    base_path: str = path_plantdoc # Deprecated
    
    # --- Output Settings ---
    output_base_dir: str = "dataset/plantdoc_ssl_dataset_linux_train_resnet10t"
    embedding_cache_dir: str = "embedding/plantdoc_embedding_train_resnet10t"
    
    # --- Out-of-Distribution / Selection Config ---
    # Thresholds are now strictly Cosine Similarity (0.0 to 1.0)
    # Suggestions:
    # 0.9+: Very high confidence (Identical to prototype)
    # 0.8+: High confidence
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "small": 0.65,  # High precision
        "medium": 0.6,
        # "large": 0.5,
    })
    
    # Balancing Strategy
    # If True, limits the number of images per class to avoid imbalance.
    do_balancing: bool = False
    # If > 0, hard limit per class. If 0, uses adaptive limit (e.g. median of counts).
    max_samples_per_class: int = 0
    
    # Max Similarity Filter (Remove very high similarity duplicates/near-copies)
    max_similarity: float = 1
    
    # --- Deduplication Settings ---
    # Hamming distance threshold for dHash (0=exact, 1-3=very similar, 5+=loose)
    dedup_hamming_threshold: int = 0

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
# 2. Model (JointMocoClassifier from eval_proposed_resnet50.py)
# -----------------------------------------------------------------------------
import copy

class JointMocoClassifier(nn.Module):
    def __init__(self, base_model_name, feature_dim, moco_k, moco_m, moco_t, num_classes):
        super().__init__()
        self.K = moco_k
        self.m = moco_m
        self.T = moco_t

        # --- Encoders ---
        self.backbone = timm.create_model(base_model_name, pretrained=False, num_classes=0)
        self.backbone_k = timm.create_model(base_model_name, pretrained=False, num_classes=0)

        # --- LẤY KÍCH THƯỚC ĐẶC TRƯNG ---
        try:
            encoder_dim = self.backbone.conv_head.out_channels
        except:
            encoder_dim = self.backbone.num_features
        
        # --- Heads ---
        self.ssl_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2), 
            nn.ReLU(), 
            nn.Linear(encoder_dim * 2, feature_dim)
        )
        self.ssl_head_k = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2), 
            nn.ReLU(), 
            nn.Linear(encoder_dim * 2, feature_dim)
        )
        
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(encoder_dim, num_classes)
        )

        # --- MoCo Initialization ---
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.ssl_head.parameters(), self.ssl_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
            
        self.register_buffer("queue", torch.randn(feature_dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.backbone.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.ssl_head.parameters(), self.ssl_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        if self.K % batch_size != 0:
            if ptr + batch_size > self.K:
                part1 = self.K - ptr
                self.queue[:, ptr:] = keys[:part1].T
                self.queue[:, :batch_size - part1] = keys[part1:].T
                ptr = batch_size - part1
            else:
                self.queue[:, ptr:ptr + batch_size] = keys.T
                ptr = (ptr + batch_size) % self.K
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def forward_ssl(self, im_q, im_k):
        q_features = self.backbone(im_q)
        q = F.normalize(self.ssl_head(q_features), dim=1)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k_features = self.backbone_k(im_k)
            k = F.normalize(self.ssl_head_k(k_features), dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(q.device)
        self._dequeue_and_enqueue(k)
        return logits, labels

    def forward_cls(self, x):
        features = self.backbone(x)
        logits = self.cls_head(features)
        return logits

    def forward(self, x, mode='ssl'):
        if mode == 'cls':
            return self.forward_cls(x)
        elif mode == 'ssl':
            features = self.backbone(x)
            return F.normalize(self.ssl_head(features), dim=1)
        return None

def load_model(cfg: Config) -> nn.Module:
    print("Initializing JointMocoClassifier architecture (main_v3 version)...")
    # Khởi tạo mô hình JointMocoClassifier
    model = JointMocoClassifier(
        base_model_name=cfg.base_model_name,
        feature_dim=cfg.feature_dim,
        moco_k=65536, moco_m=0.999, moco_t=0.07,
        num_classes=30 # Dùng dummy cho num_classes
    )
    
    # Bọc DataParallel nếu có nhiều GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    if os.path.exists(cfg.weights_path):
        try:
            state_dict = torch.load(cfg.weights_path, map_location=cfg.device)
            # Remove 'module.' prefix if weights were saved with DataParallel but loaded without it
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module.") and not isinstance(model, nn.DataParallel):
                    new_state_dict[k.replace("module.", "")] = v
                else:
                    new_state_dict[k] = v
                    
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded weights from {cfg.weights_path}")
            if missing:
                print(f"  -> Missing keys: {len(missing)}")
        except Exception as e:
            print(f"Warning: Failed to load weights ({e}).")
    else:
        print(f"Weights file not found at {cfg.weights_path}.")
        
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
def compute_embeddings(model: nn.Module, paths: List[str], cfg: Config) -> Tuple[np.ndarray, List[str]]:
    if not paths: return np.zeros((0, cfg.feature_dim), dtype=np.float32), []
    dataset = ImageListDataset(paths, EVAL_TRANSFORM)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, num_workers=0, shuffle=False)
    embs_list, out_paths = [], []
    for imgs, batch_paths in tqdm(loader, desc="Computing embeddings", leave=False):
        imgs = imgs.to(cfg.device)
        embs = model(imgs, mode='ssl').cpu().numpy()
        embs_list.append(embs)
        out_paths.extend(batch_paths)
    if not embs_list: return np.zeros((0, cfg.feature_dim), dtype=np.float32), []
    embs = np.vstack(embs_list)
    return embs.astype(np.float32), out_paths

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

def compute_dhash(image_path: str, hash_size: int = 8) -> int:
    """
    Computes Difference Hash (dHash) for an image.
    1. Resize to (hash_size + 1, hash_size).
    2. Convert to Grayscale.
    3. Compare pixels to the right.
    4. Construct binary hash.
    """
    try:
        # Open and convert to grayscale
        img = Image.open(image_path).convert("L")
        # Resize using LANCZOS for better downsampling
        img = img.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = np.array(img)
        
        # Compare pixels independent of brightness (gradient based)
        # width is hash_size + 1, so we compare col[i] > col[i+1]
        # Result is (hash_size, hash_size) boolean matrix
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # Convert to 64-bit integer
        # Flatten and pack bits
        return int(np.packbits(diff.flatten()).tobytes().hex(), 16)
    except Exception as e:
        print(f"Error computing dHash for {image_path}: {e}")
        return 0

def hamming_distance(h1: int, h2: int) -> int:
    # XOR and count set bits
    return bin(h1 ^ h2).count('1')

def remove_duplicates(embs: np.ndarray, paths: List[str], cache_dir: str, cfg: Config) -> Tuple[np.ndarray, List[str]]:
    print("Running deduplication...")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path_json = os.path.join(cache_dir, "file_hashes.json")
    hash_cache = {}
    if os.path.exists(cache_path_json):
        try:
            with open(cache_path_json, "r") as f: hash_cache = json.load(f)
        except: pass
            
    updates = 0
    
    # ---------------------------------------------------------
    # Phase 1: Exact Duplicate Removal (MD5)
    # ---------------------------------------------------------
    unique_indices_md5 = []
    seen_md5 = set()
    
    # We also need to cache dHashes for Phase 2
    path_to_dhash = {} 
    
    print(" -> Phase 1: MD5 Deduplication")
    for idx, path in enumerate(tqdm(paths, desc="Checking MD5")):
        try:
            mtime = os.path.getmtime(path)
            cached_entry = hash_cache.get(path)
            
            # Check if cache is valid (mtime match)
            if cached_entry and cached_entry.get("mtime") == mtime:
                file_md5 = cached_entry["hash"]
                # dHash might verify if we stored it, but let's compute or assume config hasn't changed
                # For simplicity, we add dHash to cache if missing or recompute
                if "dhash" in cached_entry:
                    file_dhash = cached_entry["dhash"]
                else:
                    file_dhash = compute_dhash(path)
                    cached_entry["dhash"] = file_dhash
                    updates += 1
            else:
                file_md5 = compute_file_hash(path)
                file_dhash = compute_dhash(path)
                hash_cache[path] = {"hash": file_md5, "dhash": file_dhash, "mtime": mtime}
                updates += 1
            
            # MD5 check
            if file_md5 not in seen_md5:
                seen_md5.add(file_md5)
                unique_indices_md5.append(idx)
                path_to_dhash[path] = file_dhash
                
        except Exception as e: 
            print(f"Error processing {path}: {e}")

    # Save updated cache early
    if updates > 0:
        with open(cache_path_json, "w") as f: json.dump(hash_cache, f)

    print(f"    MD5 Reduced: {len(paths)} -> {len(unique_indices_md5)}")

    # ---------------------------------------------------------
    # Phase 2: Perceptual Hashing (dHash)
    # ---------------------------------------------------------
    final_indices = []
    
    # If threshold is 0, we don't do dHash check beyond what MD5 did (assuming MD5 covers exact)
    # But dHash=0 is also exact visual match. Let's do it if threshold >= 0
    threshold = cfg.dedup_hamming_threshold
    print(f" -> Phase 2: dHash Deduplication (Threshold={threshold})")
    
    # List of kept hashes to compare against
    kept_dhashes = [] # List[int]
    
    skipped_count = 0
    for idx in tqdm(unique_indices_md5, desc="Checking dHash"):
        path = paths[idx]
        curr_dhash = path_to_dhash.get(path, 0)
        
        is_duplicate = False
        # Compare with already kept images
        # Optimization: If kept list is huge, this is slow. 
        # But for typically <50k dataset creation it's acceptable (C++ native would be better but Python is okay)
        for kept_h in kept_dhashes:
            dist = bin(curr_dhash ^ kept_h).count('1')
            if dist <= threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            final_indices.append(idx)
            kept_dhashes.append(curr_dhash)
        else:
            skipped_count += 1

    print(f"    dHash Removed: {skipped_count} near-duplicates")
    print(f"    Final Count: {len(final_indices)}")
    
    return embs[final_indices], [paths[i] for i in final_indices]

# -----------------------------------------------------------------------------
# 5. k-NN VOTING LOGIC (New)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# 5. NEAREST CENTROID & SIMILARITY LOGIC (New)
# -----------------------------------------------------------------------------
def compute_similarity_scores(
    candidate_embs: np.ndarray, 
    prototypes_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Computes the maximum cosine similarity for each candidate against ALL prototypes.
    Returns:
        max_scores: (N,) float array of best similarity scores.
        assigned_labels: (N,) string array of assigned class names.
    """
    print("\n[Similarity] Preparing prototypes...")
    
    # 1. Flatten Prototypes
    global_protos_list = []
    global_labels_list = []
    
    # prototypes_dict: {class: (M, dim)}
    for class_name, protos in prototypes_dict.items():
        # Ensure protos are 2D
        if protos.ndim == 1: protos = protos[np.newaxis, :]
        
        # Normalize prototypes just in case
        norms = np.linalg.norm(protos, axis=1, keepdims=True)
        protos = protos / (norms + 1e-9)
        
        for i in range(protos.shape[0]):
            global_protos_list.append(protos[i])
            global_labels_list.append(class_name)
            
    global_protos = np.array(global_protos_list, dtype=np.float32) # (P, dim)
    global_labels = np.array(global_labels_list) # (P,)
    
    print(f" -> Total Prototypes: {global_protos.shape[0]}")
    
    # 2. Compute Cosine Similarity (Dot Product of Normalized Vectors)
    # Candidates: (N, dim), Protos: (P, dim) -> Sim: (N, P)
    # We do this in batches to avoid OOM if N is large
    
    batch_size = 2000
    num_candidates = candidate_embs.shape[0]
    
    best_scores = np.zeros(num_candidates, dtype=np.float32)
    assigned_indices = np.zeros(num_candidates, dtype=np.int64)
    
    print(f"[Similarity] Matching {num_candidates} candidates against prototypes...")
    for start_idx in tqdm(range(0, num_candidates, batch_size), desc="Computing Similarity"):
        end_idx = min(start_idx + batch_size, num_candidates)
        batch_embs = candidate_embs[start_idx:end_idx] # (B, dim)
        
        # Dot product
        # batch_embs is already normalized in compute_embeddings
        sim_matrix = np.matmul(batch_embs, global_protos.T) # (B, P)
        
        # Find best match for each image
        # max over axis 1 (across prototypes)
        batch_best_scores = np.max(sim_matrix, axis=1)
        batch_best_idxs = np.argmax(sim_matrix, axis=1)
        
        best_scores[start_idx:end_idx] = batch_best_scores
        assigned_indices[start_idx:end_idx] = batch_best_idxs
        
    # Map indices to labels
    assigned_labels = global_labels[assigned_indices]
    
    return best_scores, assigned_labels

def balanced_selection(
    candidate_paths: List[str], 
    scores: np.ndarray, 
    labels: np.ndarray,
    threshold: float,
    cfg: Config
) -> Tuple[List[int], Dict]:
    """
    Selects images based on threshold AND performs class balancing.
    """
    # 1. Initial Filtering by Threshold & Max Similarity
    # We want: threshold < score <= max_similarity
    mask = (scores > threshold) & (scores <= cfg.max_similarity)
    valid_indices = np.where(mask)[0]
    
    if len(valid_indices) == 0:
        return [], {}
        
    print(f" -> Initial pass ({threshold} < Score <= {cfg.max_similarity}): {len(valid_indices)} images passed.")
    
    # 2. Group by Class
    class_groups = {} # {class: [(score, original_index)]}
    for idx in valid_indices:
        lbl = labels[idx]
        scr = scores[idx]
        if lbl not in class_groups:
            class_groups[lbl] = []
        class_groups[lbl].append((scr, idx))
        
    # 3. Determine Balancing Limit
    counts = [len(v) for v in class_groups.values()]
    if not counts: return [], {}
    
    if cfg.max_samples_per_class > 0:
        limit = cfg.max_samples_per_class
        mode = "Fixed"
    else:
        # Adaptive: Use Median, ensure at least some minimum
        limit = int(np.median(counts))
        limit = max(limit, 10) # Minimum safety
        mode = "Adaptive (Median)"
        
    if not cfg.do_balancing:
         limit = 999999999
         mode = "Disabled"

    print(f" -> Balancing Mode: {mode}. Limit per class: {limit}")
    
    # 4. Select Top-K per class
    final_indices = []
    distribution = {}
    
    for cls_name, items in class_groups.items():
        # Sort by score descending
        items.sort(key=lambda x: x[0], reverse=True)
        
        # Take Top-K
        selected = items[:limit]
        
        for _, original_idx in selected:
            final_indices.append(original_idx)
            
        distribution[cls_name] = len(selected)

    # Print Distribution stats
    # print(f" -> Class Distribution: {distribution}")
    
    final_indices.sort()
    return final_indices, distribution

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
    
    # Handle both string and list for robustness
    paths_to_search = cfg.ssl_data_path if isinstance(cfg.ssl_data_path, (list, tuple)) else [cfg.ssl_data_path]
    
    for path in paths_to_search:
        imgs = find_images_recursive(path)
        print(f"  - {path}: {len(imgs)} images")
        candidate_paths.extend(imgs)
        
    if not candidate_paths:
        print("No candidate images found. Exiting.")
        return

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
    embs, candidate_paths = remove_duplicates(embs, candidate_paths, cfg.embedding_cache_dir, cfg)

    # 5. Similarity Scoring (New Logic)
    print("\n[Step 5] Computing Similarity Scores (Nearest Centroid)...")
    scores, labels = compute_similarity_scores(embs, prototypes_dict)
    
    # 6. Balanced Selection
    print("\n[Step 6] Balanced Selection...")
    for name, threshold in cfg.thresholds.items():
        print(f"\ndataset '{name}' (Similarity > {threshold})")
        
        indices, dist = balanced_selection(candidate_paths, scores, labels, threshold, cfg)
        
        if indices:
            print(f" -> Selected {len(indices)} images.")
            save_subset(name, indices, candidate_paths, scores, cfg)
            
            # Save distribution report
            out_dir = os.path.join(cfg.output_base_dir, name)
            with open(os.path.join(out_dir, "distribution.json"), "w") as f:
                json.dump(dist, f, indent=2)
        else:
            print(" -> No images selected.")

if __name__ == "__main__":
    main()
