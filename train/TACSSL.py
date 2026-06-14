# %% [markdown]
# # 1. Imports & Setup
# This section imports all necessary PyTorch libraries, sets up the random seed for reproducibility, 
# and prepares the base device (usually GPU 0). The actual models will utilize DataParallel later.
# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import timm
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
import warnings
import math
from itertools import zip_longest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

warnings.filterwarnings('ignore')

# Setup seed and device
def setup_environment(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

device = setup_environment()
print(f"PyTorch: {torch.__version__}, Device: {device}, GPUs: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# %% [markdown]
# # 2. Configuration & Hyperparameters
# Please change these parameters exactly how you need them. They are centralized here for clarity.
# %%
# Training
EPOCHS = 100
WARMUP_EPOCHS = 0
ACCUMULATION_STEPS = 4

BATCH_SIZE_SSL = 256
BATCH_SIZE_CLS = 64
NUM_WORKERS = 4

LR = 3e-4
WEIGHT_DECAY = 1e-5
USE_AMP = True

# Loss weights for Joint Training
ALPHA = [0.5]  # Trọng số cho SSL loss
BETA = [1.0]   # Trọng số cho Classification loss

# Model
BASE_MODEL = ['resnet50']
FEATURE_DIM = 128 # Kích thước vector đặc trưng cho MoCo

# MoCo specific
MOCO_K = 4096*16  # Kích thước queue
MOCO_M = 0.999 # Momentum
MOCO_T = 0.07  # Temperature

# Data paths
SSL_DATA_DIR = '/kaggle/input/datasets/thuanai1/plantseg-ssl-dataset-knn-v4-v1/plantseg_ssl_dataset_knn_v4_v1/small'
CLS_TRAIN_DIR = '/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset/train'
CLS_VAL_DIR = '/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset/test'
CLS_TEST_DIR = '/kaggle/input/datasets/abdulhasibuddin/plant-doc-dataset/PlantDoc-Dataset/test'

print("=====================================PARAMETERS=====================================")
print(f"EPOCHS = {EPOCHS} | WARMUP_EPOCHS = {WARMUP_EPOCHS} | ACCUMULATION_STEPS = {ACCUMULATION_STEPS}")
print(f"BATCH_SIZE_SSL = {BATCH_SIZE_SSL} | BATCH_SIZE_CLS = {BATCH_SIZE_CLS} | NUM_WORKERS = {NUM_WORKERS}")
print(f"LR = {LR} | WEIGHT_DECAY = {WEIGHT_DECAY} | USE_AMP = {USE_AMP}")
print(f"ALPHA = {ALPHA} | BETA = {BETA}")
print(f"BASE_MODEL = {BASE_MODEL} | FEATURE_DIM = {FEATURE_DIM}")
print(f"MOCO_K = {MOCO_K} | MOCO_M = {MOCO_M} | MOCO_T = {MOCO_T}")
print("=================================================================================")

# %% [markdown]
# # 3. Data Augmentation & Loaders
# Defines the transformations for MoCo SSL and Classification, as well as loads them into
# PyTorch DataLoader wrappers with the respective batch sizes.
# %%
# 1. MoCo Augmentation
class MoCoAugmentation:
    def __init__(self, size=224):
        self.query_transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.key_transform = T.Compose([
            T.RandomResizedCrop(size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.query_transform(x), self.query_transform(x)

# 2. Classification Augmentation
train_transform = T.Compose([
    T.RandomResizedCrop(224, scale=(0.6, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Datasets
class MoCoDataset(Dataset):
    def __init__(self, root, transform):
        self.dataset = ImageFolder(root)
        self.transform = transform
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]
        q, k = self.transform(img)
        return q, k
    def __len__(self):
        return len(self.dataset)

ssl_dataset = MoCoDataset(SSL_DATA_DIR, transform=MoCoAugmentation())
train_dataset = ImageFolder(CLS_TRAIN_DIR, transform=train_transform)
val_dataset = ImageFolder(CLS_VAL_DIR, transform=eval_transform)
test_dataset = ImageFolder(CLS_TEST_DIR, transform=eval_transform)

# 4. DataLoaders
class PrefetchLoader:
    def __init__(self, loader, is_ssl=False):
        self.loader = loader
        self.is_ssl = is_ssl

    def __iter__(self):
        stream = torch.cuda.Stream()
        first = True

        for batch in self.loader:
            with torch.cuda.stream(stream):
                if self.is_ssl:
                    q, k = batch
                    next_batch = (q.to(device, non_blocking=True), k.to(device, non_blocking=True))
                else:
                    inputs, labels = batch
                    next_batch = (inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True))
                    
            if not first:
                yield current_batch
            else:
                first = False

            torch.cuda.current_stream().wait_stream(stream)
            current_batch = next_batch

        if not first:
            yield current_batch

    def __len__(self):
        return len(self.loader)

ssl_loader = DataLoader(ssl_dataset, batch_size=BATCH_SIZE_SSL, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_CLS, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_CLS, shuffle=False, num_workers=NUM_WORKERS)

# Wrap with PrefetchLoader for faster GPU data loading
ssl_loader = PrefetchLoader(ssl_loader, is_ssl=True)
train_loader = PrefetchLoader(train_loader, is_ssl=False)
val_loader = PrefetchLoader(val_loader, is_ssl=False)
test_loader = PrefetchLoader(test_loader, is_ssl=False)

num_classes = len(train_dataset.classes)
print(f"SSL Dataset: {len(ssl_dataset)} samples.")
print(f"Classification Dataset: {len(train_dataset)} train, {len(val_dataset)} validation/test samples.")
print(f"Number of classes: {num_classes}")

# %% [markdown]
# # 4. Model Architecture 
# The JointMocoClassifier architecture incorporates both self-supervised representation
# learning (MoCo) and supervised classification logic. Includes support for `nn.DataParallel`.
# %%
class JointMocoClassifier(nn.Module):
    def __init__(self, base_encoder: nn.Module, encoder_dim: int, feature_dim: int, moco_k: int, moco_m: float, moco_t: float, num_classes: int):
        super().__init__()
        self.K = moco_k
        self.m = moco_m
        self.T = moco_t

        # 1. Dependency Injection: The backbone is passed from the outside.
        self.backbone_q = base_encoder
        # We deepcopy to ensure backbone_k has the exact same architecture.
        self.backbone_k = copy.deepcopy(base_encoder)

        # 2. Heads
        self.ssl_head_q = self._build_mlp(encoder_dim, feature_dim)
        self.ssl_head_k = self._build_mlp(encoder_dim, feature_dim)
        
        self.cls_head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(encoder_dim, num_classes)
        )

        # 3. MoCo Initialization
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not updated by gradient

        for param_q, param_k in zip(self.ssl_head_q.parameters(), self.ssl_head_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Queue setup
        self.register_buffer("queue", torch.randn(feature_dim, self.K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _build_mlp(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim * 2), 
            nn.ReLU(), 
            nn.Linear(in_dim * 2, out_dim)
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder"""
        for param_q, param_k in zip(self.backbone_q.parameters(), self.backbone_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
            
        for param_q, param_k in zip(self.ssl_head_q.parameters(), self.ssl_head_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        # Ensure batch size is a divisor of K for simplicity
        if self.K % batch_size != 0:
             # simple pointer logic for non-divisible batch sizes
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

    def forward(self, im_q, im_k=None, mode='cls'):
        """
        Unified forward pass respecting standard PyTorch nn.Module contracts.
        Args:
            im_q: Input queries (or classification inputs)
            im_k: Input keys (required if mode='ssl')
            mode: 'cls' for classification, 'ssl' for MoCo contrastive step
        """
        if mode == 'cls':
            features = self.backbone_q(im_q)
            return self.cls_head(features)
            
        elif mode == 'ssl':
            if im_k is None:
                raise ValueError("im_k must be provided for ssl mode")
                
            # Process query
            q_features = self.backbone_q(im_q)
            q = F.normalize(self.ssl_head_q(q_features), dim=1)

            # Process key
            with torch.no_grad():
                self._momentum_update_key_encoder()
                k_features = self.backbone_k(im_k)
                k = F.normalize(self.ssl_head_k(k_features), dim=1)

            # Contrastive loss logits
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            logits = torch.cat([l_pos, l_neg], dim=1) / self.T
            labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device) # Fixed device dependency

            self._dequeue_and_enqueue(k)
            return logits, labels
            
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'cls' or 'ssl'.")

# %% [markdown]
# # 5. Training Loop & Evaluation
# Standard pipeline for running epochs, computing multi-objective losses, taking optimizer steps,
# evaluating validation accuracy, and plotting history. Notice that `zip_longest` handles dataloader mismatch.
# %%
def evaluate(model, dataloader, criterion):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    corrects = 0
    top5_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            # Inputs are already on device via PrefetchLoader

            with autocast(enabled=USE_AMP):
                outputs = model(inputs, mode='cls')
                loss = criterion(outputs, labels)

            probs = torch.softmax(outputs.float(), dim=1)
            
            preds = torch.argmax(probs, dim=1)
            top5 = torch.topk(probs, 5, dim=1).indices

            total_loss += loss.item() * inputs.size(0)
            corrects += (preds == labels).sum().item()
            top5_corrects += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / total
    acc = corrects / total
    top5_acc = top5_corrects / total
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    
    roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    pr_auc = np.mean([average_precision_score((all_labels == i), all_probs[:, i]) for i in range(num_classes)])

    return {
        "loss": avg_loss, "acc": acc, "top5_acc": top5_acc,
        "precision": precision, "recall": recall, "f1": f1,
        "roc_auc": roc_auc, "pr_auc": pr_auc
    }

def joint_train(model_name, model, ssl_loader, cls_loader, val_loader, epochs, alpha, beta, 
                use_amp=True, accumulation_steps=1, warmup_epochs=5):
    def calculate_class_weights(loader):
        print("Calculating class weights from training distribution...")
        class_counts = [0] * num_classes
        original_dataset = loader.dataset if hasattr(loader, 'dataset') else loader.loader.dataset
        for _, target in original_dataset.samples:
            class_counts[target] += 1
        total_samples = sum(class_counts)
        class_weights = [total_samples / count for count in class_counts]
        # Normalize weights so they sum to num_classes for stability
        weight_sum = sum(class_weights)
        class_weights = [w / weight_sum * num_classes for w in class_weights]
        return torch.FloatTensor(class_weights).to(device)

    class_weights = calculate_class_weights(cls_loader)

    ssl_criterion = nn.CrossEntropyLoss().to(device)
    # cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights).to(device)
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    
    scaler = GradScaler(enabled=use_amp)
    best_val_acc = 0.0
    history = {'train_ssl_loss': [], 'train_cls_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_ssl_loss, total_cls_loss = 0.0, 0.0
        ssl_batches, cls_batches = 0, 0
        
        current_alpha = alpha 

        if epoch < warmup_epochs:
            warmup_lr = LR * (epoch + 1) / warmup_epochs
            optimizer.param_groups[0]["lr"] = warmup_lr

        lr_backbone = optimizer.param_groups[0]['lr']
        total_steps = max(len(ssl_loader), len(cls_loader))
        
        # zip_longest allows iteration to continue until the longest dataloader finishes.
        progress_bar = tqdm(enumerate(zip_longest(ssl_loader, cls_loader)), 
                            desc=f"Epoch {epoch+1}/{epochs} (LR_bb: {lr_backbone:.2e}, α: {current_alpha:.2f})", 
                            total=total_steps)

        for i, (ssl_batch, cls_batch) in progress_bar:
            if current_alpha == 0.0 and cls_batch is None:
                progress_bar.set_postfix_str("CLS loader finished, ending epoch early.")
                break
            
            # --- Forward Pass ---
            with autocast(enabled=use_amp):
                ssl_loss = torch.tensor(0.0, device=device)
                # Ensure we only process SSL batch if it's not exhausted (not None)
                if ssl_batch is not None and current_alpha > 0:
                    im_q, im_k = ssl_batch
                    ssl_logits, ssl_labels = model(im_q, im_k=im_k, mode='ssl')
                    ssl_loss = ssl_criterion(ssl_logits, ssl_labels)
                    total_ssl_loss += ssl_loss.item() * im_q.size(0)
                    ssl_batches += im_q.size(0)

                cls_loss = torch.tensor(0.0, device=device)
                # Ensure we only process CLS batch if it's not exhausted (not None)
                if cls_batch is not None:
                    inputs, labels = cls_batch
                    cls_logits = model(inputs, mode='cls')
                    cls_loss = cls_criterion(cls_logits, labels)
                    total_cls_loss += cls_loss.item() * inputs.size(0)
                    cls_batches += inputs.size(0)
                
                combined_loss = current_alpha * ssl_loss + beta * cls_loss
                if accumulation_steps > 1:
                    combined_loss = combined_loss / accumulation_steps
            
            # --- Backward Pass ---
            if combined_loss.item() > 0:
                scaler.scale(combined_loss).backward()

            # --- Optimizer Step ---
            if (i + 1) % accumulation_steps == 0 or (i + 1) == total_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            progress_bar.set_postfix(
                ssl_loss=f"{total_ssl_loss/ssl_batches:.4f}" if ssl_batches > 0 else "---",
                cls_loss=f"{total_cls_loss/cls_batches:.4f}" if cls_batches > 0 else "---"
            )
            
        # scheduler.step()
        
        val_metrics = evaluate(model, val_loader, cls_criterion)
        avg_ssl_loss = total_ssl_loss / ssl_batches if ssl_batches > 0 else 0
        avg_cls_loss = total_cls_loss / cls_batches if cls_batches > 0 else 0
        
        history['train_ssl_loss'].append(avg_ssl_loss)
        history['train_cls_loss'].append(avg_cls_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        
        print(f"\nEpoch {epoch+1:02d} Summary | Train SSL: {avg_ssl_loss:.4f} | Train CLS: {avg_cls_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['acc']:.4f}")
        
        torch.save(model.state_dict(), f"{model_name}_last.pth")
        
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            torch.save(model.state_dict(), f"{model_name}_alpha_{alpha}_beta_{beta}_best.pth")
            print(f"🎉 New best model saved with validation accuracy: {best_val_acc:.4f}")
        
        print("-" * 50)

    return history


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    ax1.plot(history['train_ssl_loss'], label='Train SSL Loss', linestyle='--')
    ax1.plot(history['train_cls_loss'], label='Train Classification Loss', linestyle='--')
    ax1.plot(history['val_loss'], label='Validation Loss', marker='o', markersize=4)
    ax1.set_title('Loss History', fontsize=16)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['val_acc'], label='Validation Accuracy', color='green', marker='o', markersize=4)
    ax2.set_title('Validation Accuracy History', fontsize=16)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True)

    plt.suptitle('Training and Validation Metrics', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# %% [markdown]
# # 6. Main Execution
# Iterates through parameters and initiates joint training. 
# Loads the best model weight for final test-set evaluation.
# %%
if __name__ == '__main__':
    for base_model in BASE_MODEL:
        for alpha in ALPHA:
            for beta in BETA:
                print(f"=========================Experiment MODEL: {base_model} | ALPHA: {alpha} | BETA: {beta}=========================")
                
                # Setup Encoders outside the Model to preserve Modularity
                if base_model == 'resnet50':
                    base_encoder = models.resnet50(weights=ResNet50_Weights.DEFAULT)
                    # output dimension is 2048
                    encoder_dim = base_encoder.fc.in_features
                    base_encoder.fc = nn.Identity()
                else:
                    base_encoder = timm.create_model(base_model, pretrained=True, num_classes=0)
                    # Dynamically determine the output dimension of the backbone
                    try:
                        encoder_dim = base_encoder.conv_head.out_channels
                    except AttributeError:
                        encoder_dim = base_encoder.num_features
                
                # Instantiate the generalized MoCo classifier
                model = JointMocoClassifier(
                    base_encoder=base_encoder,
                    encoder_dim=encoder_dim,
                    feature_dim=FEATURE_DIM,
                    moco_k=MOCO_K,
                    moco_m=MOCO_M,
                    moco_t=MOCO_T,
                    num_classes=num_classes
                )
                
                # Wrap the ENTIRE model in DataParallel for efficient multi-GPU training
                if torch.cuda.device_count() > 1:
                    print(f"Using PyTorch DataParallel across {torch.cuda.device_count()} GPUs!")
                    model = nn.DataParallel(model)
                    
                model = model.to(device)
                
                history = joint_train(
                            base_model, model, ssl_loader, train_loader, val_loader,
                            epochs=EPOCHS,
                            alpha=alpha,
                            beta=beta,
                            use_amp=USE_AMP,
                            accumulation_steps=ACCUMULATION_STEPS,
                            warmup_epochs=WARMUP_EPOCHS,
                        )
                print("✅ Training finished!")
                
                plot_history(history)
                
                print("Loading best model for final evaluation...")
                
                # Set up evaluation model similarly
                if base_model == 'resnet50':
                    base_encoder_eval = models.resnet50(weights=None)
                    base_encoder_eval.fc = nn.Identity()
                else:
                    base_encoder_eval = timm.create_model(base_model, pretrained=False, num_classes=0)
                
                eval_model = JointMocoClassifier(
                    base_encoder=base_encoder_eval,
                    encoder_dim=encoder_dim,
                    feature_dim=FEATURE_DIM,
                    moco_k=MOCO_K, moco_m=MOCO_M, moco_t=MOCO_T,
                    num_classes=num_classes
                )
                
                if torch.cuda.device_count() > 1:
                    eval_model = nn.DataParallel(eval_model)
                    
                eval_model = eval_model.to(device)
                
                eval_model.load_state_dict(torch.load(f"{base_model}_alpha_{alpha}_beta_{beta}_best.pth"))
                
                cls_criterion = nn.CrossEntropyLoss()
                test_metrics = evaluate(eval_model, test_loader, cls_criterion)
                
                print("\n--- Final Test Results ---")
                print(f"Accuracy:    {test_metrics['acc']:.4f}")
                print(f"Top-5 Acc:   {test_metrics['top5_acc']:.4f}")
                print(f"Precision:   {test_metrics['precision']:.4f}")
                print(f"Recall:      {test_metrics['recall']:.4f}")
                print(f"F1-Score:    {test_metrics['f1']:.4f}")
                print(f"ROC-AUC:     {test_metrics['roc_auc']:.4f}")
                print(f"PR-AUC:      {test_metrics['pr_auc']:.4f}")