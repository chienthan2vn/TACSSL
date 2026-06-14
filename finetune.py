# %% [markdown]
# # PlantWild Classification Finetuning Pipeline
# This script finetunes 4 different models on the PlantWild dataset using 2x T4 GPUs.
# Formatted for Kaggle Jupyter Notebooks.

# %% [code]
import os
import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import timm
from tqdm import tqdm

# %% [markdown]
# ## 1. Centralized Configuration
# All modifiable hyperparameters and paths are centralized here.

# %% [code]
class Config:
    # Dataset Paths
    TRAIN_DIR = '/kaggle/input/datasets/thuanai1/plantseg/Plantseg/train'
    VAL_DIR = '/kaggle/input/datasets/thuanai1/plantseg/Plantseg/val'
    TEST_DIR = '/kaggle/input/datasets/thuanai1/plantseg/Plantseg/test'
    
    # Model Selection. Choose from: 'vgg16', 'vit_b_16', 'inception_resnet_v2', 'efficientnet_b0', 'resnet50', 'inception_v3', 'mobilenet_v3_large', 'densenet201', 'googlenet', 'wide_resnet50_2', 'swin_s'
    MODEL_NAME = 'googlenet'    
    # Hyperparameters
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4 # Optional customizable parameter
    
    # Device setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_GPUS = torch.cuda.device_count()

print(f"Using device: {Config.DEVICE} with {Config.NUM_GPUS} GPUs")

# %% [markdown]
# ## 2. Data Loading Pipeline
# Setup standard DataLoaders and transformations.

# %% [code]
# --- DATA TRANSFORMS ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# --- DATASETS & DATALOADERS ---
num_classes = 2 # Placeholder dummy value

try:
    image_datasets = {
        'train': datasets.ImageFolder(Config.TRAIN_DIR, data_transforms['train']),
        'val': datasets.ImageFolder(Config.VAL_DIR, data_transforms['val']),
        'test': datasets.ImageFolder(Config.TEST_DIR, data_transforms['test'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'val': DataLoader(image_datasets['val'], batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True),
        'test': DataLoader(image_datasets['test'], batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print(f"Classes: {class_names} ({num_classes} classes)")

except FileNotFoundError as e:
    # Will hit this locally before moving to Kaggle
    print(f"Warning: Kaggle Data directories not found locally. Error: {e}")
    dataloaders = {}
    dataset_sizes = {}


# %% [markdown]
# ## 3. Model Initialization
# Switchable model creation with custom classification heads. We do NOT freeze any layers to allow full fine-tuning.

# %% [code]
def initialize_model(model_name, num_classes):
    model = None
    if model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'inception_resnet_v2':
        # Load directly from timm
        model = timm.create_model('inception_resnet_v2', pretrained=True)
        num_ftrs = model.classif.in_features
        model.classif = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'inception_v3':
        model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the auxiliary net if present
        if hasattr(model, 'AuxLogits') and model.AuxLogits is not None:
            num_ftrs_aux = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
    
    elif model_name == 'densenet201':
        model = models.densenet201(weights='DEFAULT')
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'googlenet':
        model = models.googlenet(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        if hasattr(model, 'aux1') and model.aux1 is not None:
            model.aux1.fc2 = nn.Linear(model.aux1.fc2.in_features, num_classes)
        if hasattr(model, 'aux2') and model.aux2 is not None:
            model.aux2.fc2 = nn.Linear(model.aux2.fc2.in_features, num_classes)
        
    elif model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights='DEFAULT')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == 'swin_s':
        model = models.swin_s(weights='DEFAULT')
        num_ftrs = model.head.in_features
        model.head = nn.Linear(num_ftrs, num_classes)
        
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    # Ensure ALL parameters are unfrozen (requires_grad = True) for full fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    return model

# Initialize the selected model
print(f"Initializing {Config.MODEL_NAME}...")
model = initialize_model(Config.MODEL_NAME, num_classes)

# Multi-GPU support using DataParallel
if Config.NUM_GPUS > 1:
    print(f"Enabling DataParallel for {Config.NUM_GPUS} GPUs!")
    model = nn.DataParallel(model)

model = model.to(Config.DEVICE)

# Calculate Model Parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total Model Parameters: {total_params / 1e6:.2f} M")


# %% [markdown]
# ## 4. Training and Validation Pipeline
# Utilizing AdamW, CrossEntropyLoss, and recording Accuracy and Loss.

# %% [code]
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    if not dataloaders:
        print("Dataloaders missing. Skipping training module testing.")
        return model

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            pbar = tqdm(dataloaders[phase], desc=f"{phase.capitalize()} Epoch {epoch+1}/{num_epochs}", unit="batch")
            for inputs, labels in pbar:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Metrics -> Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


# %% [markdown]
# ## 5. Testing and Evaluation Pipeline
# Computes Final Accuracy, Precision, Recall, and F1-Score.

# %% [code]
def test_model(model, test_loader):
    print("--- Starting Final Evaluation on Test Set ---")
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", unit="batch"):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate evaluation metrics
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=1
        )
        
        print("\n--- Final Test Metrics ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"Params (M): {total_params / 1e6:.2f} M")


# To run training, uncomment the following line in your Kaggle notebook:
model = train_model(model, dataloaders, criterion, optimizer, num_epochs=Config.EPOCHS)

# To run final testing, uncomment the following line in your Kaggle notebook:
if 'test' in dataloaders:
    test_model(model, dataloaders['test'])
