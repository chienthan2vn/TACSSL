"""
CLAF: Contrastive Learning with Augmented Features for Imbalanced Semi-Supervised Learning
Implemented for PlantDoc Dataset using timm resnet10t backbone.

Paper: https://arxiv.org/abs/2312.09598
Dataset: PlantDoc (plant disease classification, naturally imbalanced)
"""

import os
import math
import copy
import random
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import timm
from timm.models import create_model
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report
)

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

class Config:
    # Paths
    DATA_ROOT   = "/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset"
    TRAIN_DIR   = os.path.join(DATA_ROOT, "train")
    TEST_DIR    = os.path.join(DATA_ROOT, "test")

    # Model
    BACKBONE    = "resnet10t.c3_in1k"
    PRETRAINED_CKPT = ""       # e.g. "/kaggle/input/your-pretrained-model/supcon_best.pth"
    PROJ_DIM    = 128          # projection head output dim
    FEAT_DIM    = 512          # resnet10t output channels

    # SSL ratio: fraction of train data treated as LABELED
    LABEL_RATIO = 0.2          # 20% labeled, 80% unlabeled

    # Training
    EPOCHS      = 100
    BATCH_LABELED   = 32
    BATCH_UNLABELED = 64
    LR          = 3e-4
    WEIGHT_DECAY= 1e-4
    MOMENTUM_EMA= 0.999        # EMA for momentum encoder

    # DASO / CLAF hyper-params
    CONF_THRESH = 0.95         # τ  – confidence threshold for pseudo-labels
    T_PROTO     = 0.05         # Tproto – prototype similarity temperature
    T_CONTRAST  = 0.07         # t      – contrastive loss temperature
    QUEUE_SIZE  = 128          # |Qk| per class
    LAMBDA_U    = 1.0          # weight for unsupervised loss
    LAMBDA_C    = 1.0          # weight for contrastive loss
    LAMBDA_ALIGN= 1.0          # weight for semantic alignment loss

    # Feature Augmentation
    FA_MU       = 0.8          # minimum mixture coefficient
    FA_ALPHA    = 1.0          # Beta distribution α parameter
    FA_START    = 0.80         # apply FA only in last (1-FA_START) fraction of training

    # Misc
    SEED        = 42
    NUM_WORKERS = 4
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
    CKPT_DIR    = "/kaggle/working"          # all checkpoints saved here
    SAVE_PATH   = os.path.join(CKPT_DIR, "claf_best.pth")
    LOG_PATH    = os.path.join(CKPT_DIR, "claf_metrics.csv")


# ──────────────────────────────────────────────
# TRANSFORMS
# ──────────────────────────────────────────────

def get_transforms(img_size=224):
    normalize = T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])

    # Weak augmentation (Aw)
    weak = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    # Strong augmentation (As) – RandAugment style
    strong = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ColorJitter(0.4, 0.4, 0.4, 0.1),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        normalize,
    ])

    # Eval / test
    val = T.Compose([
        T.Resize(int(img_size * 1.14)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        normalize,
    ])

    return weak, strong, val


# ──────────────────────────────────────────────
# DATASETS
# ──────────────────────────────────────────────

class LabeledDataset(Dataset):
    """Wraps an ImageFolder subset, returns (weak_aug_img, label)."""

    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label


class UnlabeledDataset(Dataset):
    """Returns (weak_aug, strong_aug) for unlabeled samples."""

    def __init__(self, dataset, indices, weak_tf, strong_tf):
        self.dataset   = dataset
        self.indices   = indices
        self.weak_tf   = weak_tf
        self.strong_tf = strong_tf

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, _ = self.dataset[self.indices[i]]
        return self.weak_tf(img), self.strong_tf(img)


def build_datasets(cfg: Config):
    """Split train set into labeled / unlabeled respecting class imbalance."""
    weak_tf, strong_tf, val_tf = get_transforms()

    # Raw datasets (PIL images, no transform)
    raw_train = ImageFolder(cfg.TRAIN_DIR)
    raw_test  = ImageFolder(cfg.TEST_DIR)

    num_classes = len(raw_train.classes)
    targets     = np.array(raw_train.targets)

    labeled_idx   = []
    unlabeled_idx = []

    for cls_id in range(num_classes):
        cls_indices = np.where(targets == cls_id)[0].tolist()
        random.shuffle(cls_indices)
        n_label = max(1, int(len(cls_indices) * cfg.LABEL_RATIO))
        labeled_idx.extend(cls_indices[:n_label])
        unlabeled_idx.extend(cls_indices[n_label:])

    labeled_ds   = LabeledDataset(raw_train, labeled_idx, weak_tf)
    unlabeled_ds = UnlabeledDataset(raw_train, unlabeled_idx, weak_tf, strong_tf)
    test_ds      = ImageFolder(cfg.TEST_DIR, transform=val_tf)

    # Class sample counts for FA probability (Eq. 4)
    class_counts = Counter(targets[labeled_idx])
    Nk = torch.zeros(num_classes)
    for k, cnt in class_counts.items():
        Nk[k] = cnt

    print(f"  Classes : {num_classes}")
    print(f"  Labeled : {len(labeled_idx)}")
    print(f"  Unlabeled: {len(unlabeled_idx)}")
    print(f"  Test    : {len(test_ds)}")
    print(f"  Class counts (labeled): {sorted(class_counts.items())[:5]} ...")

    return labeled_ds, unlabeled_ds, test_ds, num_classes, Nk


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """2-layer MLP projection head (SimCLR style)."""

    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class CLAFModel(nn.Module):
    """
    Online encoder f_θ  + linear classifier + projection head.
    Momentum encoder f_θ' is maintained externally (EMA copy).
    """

    def __init__(self, backbone_name, num_classes, feat_dim, proj_dim):
        super().__init__()
        self.encoder    = create_model(backbone_name, pretrained=True, num_classes=0)
        
        self.classifier = nn.Linear(feat_dim, num_classes)
        # Khởi tạo trọng số nhỏ để bảo vệ pretrained weights của encoder khỏi bị phá huỷ ở batch đầu tiên
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
        
        self.projector  = ProjectionHead(feat_dim, hidden_dim=feat_dim, out_dim=proj_dim)

    def forward(self, x):
        z = self.encoder(x)           # [B, feat_dim]
        logits = self.classifier(z)   # [B, K]
        return z, logits

    def project(self, z):
        return self.projector(z)      # [B, proj_dim]


# ──────────────────────────────────────────────
# QUEUES  (feature queue Q and embedding queue E)
# ──────────────────────────────────────────────

class ClassQueue:
    """Fixed-size FIFO queue per class for features (Q) or embeddings (E)."""

    def __init__(self, num_classes, queue_size, feat_dim, device):
        self.num_classes = num_classes
        self.queue_size  = queue_size
        self.device      = device
        # [K, queue_size, D]
        self.data    = torch.zeros(num_classes, queue_size, feat_dim, device=device)
        self.ptr     = torch.zeros(num_classes, dtype=torch.long, device=device)
        self.filled  = torch.zeros(num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def enqueue(self, features, labels):
        """features: [B, D], labels: [B] (LongTensor)"""
        for feat, lbl in zip(features, labels):
            k   = lbl.item()
            ptr = self.ptr[k].item()
            self.data[k, ptr] = feat
            self.ptr[k]       = (ptr + 1) % self.queue_size
            self.filled[k]    = min(self.filled[k] + 1, self.queue_size)

    def get(self, cls_id):
        """Return valid stored entries for class cls_id."""
        n = self.filled[cls_id].item()
        if n == 0:
            return None
        return self.data[cls_id, :n]   # [n, D]

    def prototypes(self):
        """Return per-class mean feature (prototype). [K, D]"""
        protos = []
        for k in range(self.num_classes):
            n = self.filled[k].item()
            if n > 0:
                protos.append(self.data[k, :n].mean(0))
            else:
                protos.append(torch.zeros(self.data.shape[-1], device=self.device))
        return torch.stack(protos)  # [K, D]


# ──────────────────────────────────────────────
# FEATURE AUGMENTATION  (Section 4.1)
# ──────────────────────────────────────────────

def class_fa_probs(Nk: torch.Tensor):
    """
    Eq. 4: P_k = (N1 - Nk) / N1
    Nk: [K] labeled sample counts per class
    """
    N1 = Nk.max()
    return ((N1 - Nk) / N1).clamp(0, 1)


@torch.no_grad()
def feature_augmentation(z_labeled, y_labeled, z_unlabeled_w,
                          fa_probs, alpha=1.0, mu=0.8, device="cpu"):
    """
    For each labeled sample from class k, augment with probability P_k by
    mixing its feature with a randomly sampled unlabeled feature.

    Vectorized implementation:
      - Unlabeled partner indices sampled in one shot via torch.randint (GPU-friendly).
      - Lambda values sampled in bulk from Beta, then clamped to satisfy label validity.
      - Python loop retained only for the per-sample Bernoulli gate (fa_probs[k]),
        which is negligible vs. the tensor ops and keeps the mask logic readable.

    Returns:
        z_aug   : [B_aug, D]   augmented features  (None if no sample passes gate)
        y_aug   : [B_aug]      preserved labels
        lam_aug : [B_aug]      mixture coefficients (for confidence vector v)
    """
    B_l = z_labeled.size(0)
    B_u = z_unlabeled_w.size(0)

    # ── vectorized sampling ──────────────────────────────────────────────────
    # Sample one unlabeled partner for every labeled sample upfront (cheap to discard)
    rand_idx  = torch.randint(0, B_u, (B_l,), device=device)          # [B_l]
    z_u_sampled = z_unlabeled_w[rand_idx]                              # [B_l, D]

    # Sample lambda in bulk; enforce label-validity constraint (Eq. 3 discussion)
    lam_raw = torch.tensor(
        np.random.beta(alpha, alpha, size=B_l), dtype=torch.float32, device=device
    )
    lam = torch.clamp(torch.max(lam_raw, 1 - lam_raw), min=mu)        # [B_l]

    # ── per-sample Bernoulli gate based on class-dependent probability ───────
    keep_mask = torch.zeros(B_l, dtype=torch.bool, device=device)
    for i in range(B_l):
        k = y_labeled[i].item()
        if random.random() < fa_probs[k].item():
            keep_mask[i] = True

    if not keep_mask.any():
        return None, None, None

    # ── apply gate and mix ───────────────────────────────────────────────────
    z_l_k   = z_labeled[keep_mask]                                     # [B_aug, D]
    z_u_k   = z_u_sampled[keep_mask]
    lam_k   = lam[keep_mask].unsqueeze(-1)                             # [B_aug, 1]

    z_aug   = lam_k * z_l_k + (1 - lam_k) * z_u_k                    # [B_aug, D]
    y_aug   = y_labeled[keep_mask]                                     # [B_aug]
    lam_aug = lam_k.squeeze(-1)                                        # [B_aug]

    return z_aug, y_aug, lam_aug


# ──────────────────────────────────────────────
# CONTRASTIVE LOSS  (Section 4.2 – Eq. 7-8)
# ──────────────────────────────────────────────

def claf_contrastive_loss(e_s, pseudo_labels, conf_scores,
                          emb_queue: ClassQueue, label_conf_queue: ClassQueue,
                          temperature=0.07):
    """
    e_s          : [B_u, D] embeddings of strongly-augmented unlabeled samples
    pseudo_labels: [B_u]    predicted class indices
    conf_scores  : [B_u]    max pseudo-label probability (s_i)
    emb_queue    : embedding queue E (per-class embeddings of labeled+augmented)
    label_conf_queue: stores λ-weights for augmented, 1.0 for labeled
    temperature  : t
    """
    B    = e_s.size(0)
    K    = emb_queue.num_classes
    loss = torch.tensor(0.0, device=e_s.device, requires_grad=True)
    cnt  = 0

    # Collect ALL class embeddings for denominator: [N_total, D]
    all_emb_list = []
    for k in range(K):
        e_k = emb_queue.get(k)
        if e_k is not None:
            all_emb_list.append(e_k)
    if len(all_emb_list) == 0:
        return loss
    all_emb = torch.cat(all_emb_list, dim=0)   # [N_total, D]

    for i in range(B):
        si   = conf_scores[i]
        if si == 0:
            continue
        pi   = pseudo_labels[i].item()
        e_p  = emb_queue.get(pi)                # [n_pi, D] positives
        if e_p is None or e_p.size(0) == 0:
            continue

        # Label-confidence vector v for positive embeddings
        v_p  = label_conf_queue.get(pi)         # [n_pi, 1] or None
        if v_p is None:
            v_p = torch.ones(e_p.size(0), 1, device=e_s.device)

        # Weights w_ip = s_i * v_j  (Eq. before Eq.7)
        w_ip = (si * v_p.squeeze(-1)).detach()  # [n_pi]

        # Numerator: sim(e_s_i, e_p) / t
        sim_pos  = (e_s[i].unsqueeze(0) @ e_p.T) / temperature   # [1, n_pi]
        # Denominator: sum over all classes
        sim_all  = (e_s[i].unsqueeze(0) @ all_emb.T) / temperature  # [1, N_total]
        log_denom = torch.logsumexp(sim_all, dim=1)                  # [1]

        # Weighted per-positive loss  (Eq. 8)
        n_pi = e_p.size(0)
        lc_i = -(w_ip * (sim_pos.squeeze(0) - log_denom)).sum() / n_pi
        loss = loss + lc_i
        cnt  += 1

    return loss / max(cnt, 1)


# ──────────────────────────────────────────────
# SEMANTIC ALIGNMENT LOSS  (from DASO)
# ──────────────────────────────────────────────

def semantic_alignment_loss(q_sem, p_hat):
    """
    Encourage consistency between linear pseudo-label p_hat and
    similarity-based semantic pseudo-label q_sem.
    Simple KL: KL(q_sem || p_hat)
    """
    q = q_sem.detach()
    p = F.log_softmax(p_hat, dim=-1)
    return F.kl_div(p, q, reduction="batchmean")


# ──────────────────────────────────────────────
# EMA  (momentum encoder)
# ──────────────────────────────────────────────

@torch.no_grad()
def update_ema(online: nn.Module, target: nn.Module, momentum: float):
    # Update parameters
    for p_o, p_t in zip(online.parameters(), target.parameters()):
        p_t.data.mul_(momentum).add_(p_o.data, alpha=1 - momentum)
    # Update buffers (like BatchNorm running stats)
    for b_o, b_t in zip(online.buffers(), target.buffers()):
        b_t.data.copy_(b_o.data)


# ──────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────

class CLAFTrainer:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        torch.manual_seed(cfg.SEED)
        random.seed(cfg.SEED)
        np.random.seed(cfg.SEED)

        # ── Data ──
        (self.labeled_ds, self.unlabeled_ds,
         self.test_ds, self.num_classes, Nk) = build_datasets(cfg)

        self.fa_probs = class_fa_probs(Nk).to(cfg.DEVICE)

        self.labeled_loader = DataLoader(
            self.labeled_ds, batch_size=cfg.BATCH_LABELED,
            shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=True)

        self.unlabeled_loader = DataLoader(
            self.unlabeled_ds, batch_size=cfg.BATCH_UNLABELED,
            shuffle=True, num_workers=cfg.NUM_WORKERS, drop_last=True, pin_memory=True)

        self.test_loader = DataLoader(
            self.test_ds, batch_size=64, shuffle=False,
            num_workers=cfg.NUM_WORKERS, pin_memory=True)

        # ── Model ──
        self.model = CLAFModel(
            cfg.BACKBONE, self.num_classes, cfg.FEAT_DIM, cfg.PROJ_DIM
        ).to(cfg.DEVICE)

        # Load custom pretrained backbone weights if provided (e.g. SSL pretrained)
        if getattr(cfg, "PRETRAINED_CKPT", "") and os.path.isfile(cfg.PRETRAINED_CKPT):
            print(f"🔄 Loading custom pretrained weights from: {cfg.PRETRAINED_CKPT}")
            ckpt = torch.load(cfg.PRETRAINED_CKPT, map_location=cfg.DEVICE)
            state_dict = ckpt.get("model", ckpt.get("model_state", ckpt.get("state_dict", ckpt)))
            
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                # If the checkpoint is from SupCon/MoCo, it might have 'encoder.' prefix
                if k.startswith("encoder."):
                    new_state_dict[k.replace("encoder.", "")] = v
                else:
                    new_state_dict[k] = v
                    
            msg = self.model.encoder.load_state_dict(new_state_dict, strict=False)
            print(f"  └─ Missing keys: {len(msg.missing_keys)} | Unexpected keys: {len(msg.unexpected_keys)}")

        # Momentum encoder (EMA copy, no grad)
        self.ema_model = copy.deepcopy(self.model).to(cfg.DEVICE)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_model.eval()

        # ── Queues ──
        self.feat_queue  = ClassQueue(
            self.num_classes, cfg.QUEUE_SIZE, cfg.FEAT_DIM, cfg.DEVICE)
        self.emb_queue   = ClassQueue(
            self.num_classes, cfg.QUEUE_SIZE, cfg.PROJ_DIM, cfg.DEVICE)
        # Label-confidence queue: stores λ (scalar) – we reuse ClassQueue with dim=1
        self.lconf_queue = ClassQueue(
            self.num_classes, cfg.QUEUE_SIZE, 1, cfg.DEVICE)

        # ── Optimiser ──
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.EPOCHS)

        self.best_acc  = 0.0
        self.fa_active = False   # will be activated in last (1-FA_START) epochs

    # ─── one training epoch ───────────────────

    def _train_epoch(self, epoch):
        self.model.train()
        cfg   = self.cfg
        dev   = cfg.DEVICE
        # Drive loop by unlabeled data – labeled iterator wraps around automatically.
        # Using min() would discard the majority of unlabeled batches each epoch.
        total = len(self.unlabeled_loader)

        labeled_iter   = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)

        metrics = dict(loss=0, loss_cls=0, loss_u=0, loss_c=0, loss_align=0)

        pbar = tqdm(range(total), desc=f"Epoch {epoch:3d}/{cfg.EPOCHS} [Train]", leave=False)
        for step in pbar:
            # ── fetch batches ──
            try:
                x_l, y_l = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                x_l, y_l = next(labeled_iter)

            try:
                x_u_w, x_u_s = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                x_u_w, x_u_s = next(unlabeled_iter)

            x_l, y_l   = x_l.to(dev), y_l.to(dev)
            x_u_w, x_u_s = x_u_w.to(dev), x_u_s.to(dev)

            # ───────────────────────────────────
            # A. Momentum encoder (no_grad): labeled + unlabeled weak
            # ───────────────────────────────────
            with torch.no_grad():
                z_l_ema, _       = self.ema_model(x_l)   # [B_l, D]
                z_u_w_ema, _     = self.ema_model(x_u_w) # [B_u, D]

                # Semantic pseudo-label via prototype similarity (Eq. 2)
                protos           = self.feat_queue.prototypes()  # [K, D]
                sim_u_w          = F.normalize(z_u_w_ema, dim=-1) @ \
                                   F.normalize(protos, dim=-1).T   # [B_u, K]
                q_hat            = sim_u_w / cfg.T_PROTO            # logits
                q_sem            = torch.softmax(q_hat, dim=-1)    # semantic pseudo-label

            # ───────────────────────────────────
            # B & C. Online encoder: Single forward pass for all inputs
            # ───────────────────────────────────
            # Concatenate to stabilize BatchNorm statistics (standard FixMatch practice).
            # If passed sequentially, the strong augmentations (x_u_s) would skew the 
            # running BN stats, ruining the ema_model's representations.
            B_l, B_u = x_l.size(0), x_u_w.size(0)
            x_all = torch.cat([x_l, x_u_w, x_u_s], dim=0)
            z_all, logits_all = self.model(x_all)

            z_l, logits_l = z_all[:B_l], logits_all[:B_l]
            loss_cls = F.cross_entropy(logits_l, y_l)

            z_u_w_online, logits_u_w = z_all[B_l:B_l+B_u], logits_all[B_l:B_l+B_u]
            z_u_s_online, logits_u_s = z_all[B_l+B_u:], logits_all[B_l+B_u:]

            p_hat    = torch.softmax(logits_u_w, dim=-1)   # [B_u, K]

            # Fused pseudo-label (DASO-style blending)
            p_fused  = 0.5 * p_hat + 0.5 * q_sem
            conf_f, pl_f = p_fused.max(dim=-1)

            # Confidence mask
            mask     = (conf_f >= cfg.CONF_THRESH).float()

            # Semi-supervised (consistency) loss – FixMatch-like
            # Uses cached logits_u_s instead of a second self.model(x_u_s) call.
            loss_u   = (F.cross_entropy(
                            logits_u_s,
                            pl_f, reduction="none") * mask).mean()

            # Semantic alignment loss (Lalign)
            loss_align = semantic_alignment_loss(q_sem, logits_u_w)

            # ───────────────────────────────────
            # D. Feature Augmentation (FA) – last FA_START% of training
            # ───────────────────────────────────
            with torch.no_grad():
                if self.fa_active:
                    z_aug, y_aug, lam_aug = feature_augmentation(
                        z_l_ema, y_l, z_u_w_ema,
                        self.fa_probs, alpha=cfg.FA_ALPHA,
                        mu=cfg.FA_MU, device=dev)
                else:
                    z_aug, y_aug, lam_aug = None, None, None

                # Update feature queue Q with labeled + augmented features
                self.feat_queue.enqueue(z_l_ema, y_l)
                if z_aug is not None:
                    self.feat_queue.enqueue(z_aug, y_aug)

            # ───────────────────────────────────
            # E. Build embedding queue E
            # ───────────────────────────────────
            with torch.no_grad():
                e_l   = self.ema_model.project(z_l_ema)             # [B_l, proj]
                self.emb_queue.enqueue(e_l, y_l)
                # lconf = 1 for labeled
                lconf_l = torch.ones(e_l.size(0), 1, device=dev)
                self.lconf_queue.enqueue(lconf_l, y_l)

                if z_aug is not None:
                    e_aug = self.ema_model.project(z_aug)
                    self.emb_queue.enqueue(e_aug, y_aug)
                    lconf_a = lam_aug.unsqueeze(-1)
                    self.lconf_queue.enqueue(lconf_a, y_aug)

            # ───────────────────────────────────
            # F. Contrastive loss Lc
            # ───────────────────────────────────
            e_s_proj = self.model.project(z_u_s_online)  # embeddings for strong

            # Confidence vector s (Eq. 5)
            s_vec = torch.where(conf_f >= cfg.CONF_THRESH, conf_f,
                                torch.zeros_like(conf_f))

            # e_s_proj retains grad so loss_c trains the encoder directly (Eq. 7-8).
            # The paper does not specify stop-gradient on e_i^(s); keeping gradient
            # flow here is the correct and more powerful formulation.
            loss_c = claf_contrastive_loss(
                e_s_proj,
                pl_f, s_vec,
                self.emb_queue, self.lconf_queue,
                temperature=cfg.T_CONTRAST)

            # ───────────────────────────────────
            # G. Total loss (Eq. 9)
            # ───────────────────────────────────
            loss = (loss_cls
                    + cfg.LAMBDA_U     * loss_u
                    + cfg.LAMBDA_ALIGN * loss_align
                    + cfg.LAMBDA_C     * loss_c)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            # Update momentum encoder
            update_ema(self.model, self.ema_model, cfg.MOMENTUM_EMA)

            # Accumulate metrics
            metrics["loss"]       += loss.item()
            metrics["loss_cls"]   += loss_cls.item()
            metrics["loss_u"]     += loss_u.item()
            metrics["loss_c"]     += loss_c.item() if isinstance(loss_c, torch.Tensor) else 0
            metrics["loss_align"] += loss_align.item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{loss_cls.item():.4f}"
            })

        n = max(total, 1)
        return {k: v / n for k, v in metrics.items()}

    # ─── evaluation ───────────────────────────

    @torch.no_grad()
    def evaluate(self, verbose=False):
        """
        Evaluate on test set using the EMA model.
        """
        self.ema_model.eval()
        all_preds  = []
        all_labels = []

        pbar = tqdm(self.test_loader, desc="Evaluating", leave=False)
        for x, y in pbar:
            x = x.to(self.cfg.DEVICE)
            _, logits = self.ema_model(x)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

        all_preds  = np.array(all_preds)
        all_labels = np.array(all_labels)

        # zero_division=0 avoids warnings for classes absent in predictions
        acc        = accuracy_score(all_labels, all_preds) * 100
        f1_mac     = f1_score(all_labels, all_preds, average="macro",    zero_division=0) * 100
        f1_wt      = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
        prec       = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
        rec        = recall_score(all_labels, all_preds, average="macro",    zero_division=0) * 100

        return dict(accuracy=acc, f1_macro=f1_mac, f1_weighted=f1_wt,
                    precision=prec, recall=rec)

    def _init_csv_log(self):
        """Write CSV header for metric log."""
        os.makedirs(self.cfg.CKPT_DIR, exist_ok=True)
        with open(self.cfg.LOG_PATH, "w") as f:
            f.write("epoch,loss,loss_cls,loss_u,loss_c,loss_align,"
                    "accuracy,f1_macro,f1_weighted,precision,recall\n")

    def _append_csv_log(self, epoch, train_metrics, eval_metrics):
        """Append one row to the CSV metric log."""
        with open(self.cfg.LOG_PATH, "a") as f:
            f.write(
                f"{epoch},"
                f"{train_metrics['loss']:.6f},"
                f"{train_metrics['loss_cls']:.6f},"
                f"{train_metrics['loss_u']:.6f},"
                f"{train_metrics['loss_c']:.6f},"
                f"{train_metrics['loss_align']:.6f},"
                f"{eval_metrics['accuracy']:.4f},"
                f"{eval_metrics['f1_macro']:.4f},"
                f"{eval_metrics['f1_weighted']:.4f},"
                f"{eval_metrics['precision']:.4f},"
                f"{eval_metrics['recall']:.4f}\n"
            )

    # ─── main train loop ──────────────────────

    def train(self):
        cfg            = self.cfg
        fa_start_epoch = int(cfg.EPOCHS * cfg.FA_START)
        best_f1        = 0.0          # model selection criterion: macro-F1

        os.makedirs(cfg.CKPT_DIR, exist_ok=True)
        self._init_csv_log()

        print(f"\n🚀 Starting training for {cfg.EPOCHS} epochs...\n" + "═" * 80)

        for epoch in range(1, cfg.EPOCHS + 1):
            # Activate feature augmentation in last (1-FA_START) fraction of training
            if epoch == fa_start_epoch + 1:
                self.fa_active = True
                print(f"  🌟 Feature Augmentation activated at epoch {epoch} 🌟")

            train_m = self._train_epoch(epoch)
            self.scheduler.step()

            # Full evaluation – verbose per-class report at first, last and best epochs
            verbose = (epoch == 1 or epoch == cfg.EPOCHS)
            eval_m  = self.evaluate(verbose=verbose)

            is_best = eval_m["f1_macro"] > best_f1
            if is_best:
                best_f1 = eval_m["f1_macro"]

            # ── Console row ──────────────────────────────────────────────────
            marker = " ✨ [NEW BEST]" if is_best else ""
            
            print(f"📊 Epoch {epoch:3d}/{cfg.EPOCHS} Summary:")
            print(f"   ├─ Train Loss : {train_m['loss']:.4f} (Cls: {train_m['loss_cls']:.4f} | U: {train_m['loss_u']:.4f} | C: {train_m['loss_c']:.4f} | Align: {train_m['loss_align']:.4f})")
            print(f"   └─ Val Metrics: Acc: {eval_m['accuracy']:5.2f}% | F1-Mac: {eval_m['f1_macro']:5.2f}% | F1-Wt: {eval_m['f1_weighted']:5.2f}% | Prec: {eval_m['precision']:5.2f}% | Rec: {eval_m['recall']:5.2f}%{marker}")
            print("─" * 80)

            # ── CSV log ──────────────────────────────────────────────────────
            self._append_csv_log(epoch, train_m, eval_m)

            # ── Checkpoint ───────────────────────────────────────────────────
            # Save every epoch (latest) and separately keep the best F1 checkpoint.
            ckpt = {
                "epoch":           epoch,
                "model_state":     self.model.state_dict(),
                "ema_model_state": self.ema_model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "metrics":         eval_m,
            }
            torch.save(ckpt, os.path.join(cfg.CKPT_DIR, "claf_latest.pth"))

            if is_best:
                torch.save(ckpt, cfg.SAVE_PATH)   # claf_best.pth
                # Also print full per-class report when a new best is found
                if not verbose:
                    self.evaluate(verbose=True)

        print(f"\n🎯 Training complete.")
        print(f"   🏆 Best macro-F1 : {best_f1:.2f}%")
        print(f"   💾 Best ckpt     : {cfg.SAVE_PATH}")
        print(f"  Latest ckpt   : {os.path.join(cfg.CKPT_DIR, 'claf_latest.pth')}")
        print(f"  Metric log    : {cfg.LOG_PATH}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    cfg     = Config()
    trainer = CLAFTrainer(cfg)
    trainer.train()