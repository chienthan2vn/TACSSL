import os
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
from sklearn.cluster import KMeans
from kneed import KneeLocator
import sys

# Add current directory to path to allow importing main (for config/model if needed, 
# although ideally we keep them decoupled or share a common config file)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from main import CONFIG, load_model, compute_embeddings, find_images_recursive

# -----------------------------------------------------------------------------
# Clustering Utilities
# -----------------------------------------------------------------------------
def get_optimal_k(data: np.ndarray, max_k: int = 10, random_state: int = 42) -> int:
    """
    Determine optimal k for K-Means using the Elbow Method.
    """
    n_samples = data.shape[0]
    if n_samples <= 1:
        return 1
    
    # K cannot be greater than number of samples
    limit_k = min(n_samples, max_k)
    if limit_k < 2:
        return 1

    inertias = []
    # If limit is small, allow full range
    k_values = range(1, limit_k + 1)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init='auto')
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        
    # Use KneeLocator to find the elbow point
    kn = KneeLocator(list(k_values), inertias, curve='convex', direction='decreasing')
    
    elbow_k = kn.knee
    
    if elbow_k is None:
        print(f"Warning: Elbow not found, defaulting to k={limit_k}")
        return limit_k

    # Ensure optimal_k does not exceed total samples
    optimal_k = min(n_samples, int(elbow_k))

    print(f"  Elbow found at {elbow_k}. Scaling to k={optimal_k}")
    return optimal_k

def compute_class_prototypes(
    model, 
    class_name: str, 
    image_paths: List[str], 
    cfg,
    max_k: int = 15
) -> np.ndarray:
    """
    Compute embeddings for a class, run KMeans, and return centroids (prototypes).
    """
    if not image_paths:
        return np.zeros((0, cfg.feature_dim))

    pass_cfg = cfg # Use passed config

    embeddings, _ = compute_embeddings(model, image_paths, pass_cfg)
    
    if len(embeddings) == 0:
        return np.zeros((0, cfg.feature_dim))
        
    # Find Optimal K
    best_k = get_optimal_k(embeddings, max_k=max_k)
    
    # Run Final KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
    kmeans.fit(embeddings)
    centroids = kmeans.cluster_centers_
    
    # Re-normalize centroids (since we work with cosine similarity)
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / (norms + 1e-9)
    
    print(f"  Class '{class_name}': Found {best_k} prototypes (from {len(embeddings)} images).")
    return centroids.astype(np.float32)

# -----------------------------------------------------------------------------
# Main Prototype Generation Function
# -----------------------------------------------------------------------------
def get_all_prototypes(cfg, force_recompute: bool = False) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Generate prototypes for all classes in the configured source directories.
    Returns:
        prototypes_dict: {class_name: np.array(n_protos, dim)}
        all_class_names: sorted list of class names
    """
    cache_path = os.path.join(cfg.embedding_cache_dir, "prototypes_clustered.npz")
    
    # Try Loading Cache
    if not force_recompute and os.path.exists(cache_path):
        print(f"Loading cached prototypes from {cache_path}")
        try:
            data = np.load(cache_path, allow_pickle=True)
            prototypes_dict = {k: v for k, v in data.items() if k != "class_names"}
            all_class_names = data.get("class_names", sorted(list(prototypes_dict.keys()))).tolist()
            return prototypes_dict, all_class_names
        except Exception as e:
            print(f"Failed to load cache ({e}), recomputing...")

    # Recompute
    print("Computing new prototypes with K-Means clustering...")
    model = load_model(cfg)
    
    prototypes_dict = {}
    class_to_paths = {}

    # Aggregate images from all source paths
    base_path = getattr(cfg, 'base_path', '')
    # User requested: source_paths are subfolders within base_path
    if os.path.exists(base_path):
        source_paths = [os.path.join(base_path, d) for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    else:
        source_paths = []
        print(f"Warning: Base path {base_path} not found.")

    print(f"Scanning {len(source_paths)} source directories from {base_path}...")
    for root_dir in source_paths:
        if not os.path.exists(root_dir):
            print(f"  Warning: Source directory {root_dir} does not exist. Skipping.")
            continue
        
        # Check subdirectories to see if they act as class names
        subfolders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        # Strategy 1: The directory contains class subfolders
        if len(subfolders) > 0:
            for class_name in subfolders:
                class_dir = os.path.join(root_dir, class_name)
                paths = find_images_recursive(class_dir)
                
                if class_name not in class_to_paths:
                    class_to_paths[class_name] = []
                class_to_paths[class_name].extend(paths)
        
        # Strategy 2: The directory IS the class folder (contains images directly)
        # We do this if we didn't populate above, or just to be robust. 
        # However, purely checking 'else' might miss mixed content.
        # Given the context (PlantDoc), these ARE the class folders.
        else:
            paths = find_images_recursive(root_dir)
            if len(paths) > 0:
                class_name = os.path.basename(root_dir)
                if class_name not in class_to_paths:
                    class_to_paths[class_name] = []
                class_to_paths[class_name].extend(paths)

    print(f"Found {len(class_to_paths)} unique classes.")

    # Compute prototypes for each consolidated class
    for class_name, paths in tqdm(sorted(class_to_paths.items()), desc="Clustering Prototypes"):
        # Increase max_k for search space
        centroids = compute_class_prototypes(model, class_name, paths, cfg, max_k=15)
        if len(centroids) > 0:
            prototypes_dict[class_name] = centroids

    # Save Cache
    os.makedirs(cfg.embedding_cache_dir, exist_ok=True)
    save_dict = {k: v for k, v in prototypes_dict.items()}
    save_dict["class_names"] = np.array(sorted(prototypes_dict.keys()))
    np.savez_compressed(cache_path, **save_dict)
    
    print(f"Saved clustered prototypes to {cache_path}")
    return prototypes_dict, sorted(prototypes_dict.keys())

if __name__ == "__main__":
    from main import CONFIG
    # Force recompute to test new K logic
    protos, names = get_all_prototypes(CONFIG, force_recompute=True)
    print("Test Complete.")
