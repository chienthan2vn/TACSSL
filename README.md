# TACSSL: Task-Aware Contrastive Semi-Supervised Learning

## Abstract
Plant disease detection is crucial for disease management and maintaining agricultural productivity. Supervised deep learning methods for image classification have shown their strong performance in solving this task. However, training such models requires large labeled datasets, which are laborious and expensive to obtain. To reduce the dependency on labeled data, recent studies have explored semi-supervised learning (SSL) to leverage unlabeled data. Existing SSL models typically consist of a self-supervised learning step on unlabeled data followed by a fine-tuning step on labeled data. In such a design, the first step learns general, task-agnostic representations, thus limits their effectiveness in low-label settings. In this paper, we propose Task-Aware Contrastive Semi-Supervised Learning (TACSSL) framework that performs joint self-supervised contrastive and supervised learning in one step. This design encourages the model to learn task-specific features that are not only discriminative in general but also closely aligned with disease categories. In addition, to reduce the harmful effect of noisy unlabeled data and retain only data relevant to the task, we introduce a data selection step based on semantic closeness between images. Experimental results on PlantDoc, PlantWild, and PlantSeg datasets, show that TACSSL achieved strong and consistent performance across different data conditions, outperforming other SSL methods. Using relatively lightweight ResNet50 as the backbone, TACSSL achieved 72.88% Accuracy and 73.77% F1-score on PlantDoc. Similar results were observed on more challenging PlantWild, where TACSSL achieved 68.59% Accuracy and 65.71% F1-score, and on PlantSeg, where it reached 75.03% Accuracy and 67.01% F1-score, demonstrating the effectiveness and robustness of the joint learning framework across datasets of varying complexity.

## Project Structure
```text
TACSSL/
├── compare/                 # Scripts and modules for comparing with other SSL methods
├── create_dataset/          # Scripts for dataset preparation and selection
├── train/
│   └── TACSSL.py            # Main training script for the TACSSL framework
├── finetune.py              # Script for fine-tuning models
├── plot_lambda_metrics.py   # Utility script for plotting evaluation metrics
└── README.md                # Project documentation
```

## Quick Start
1. **Dataset Preparation**: Prepare your datasets using the tools provided in the `create_dataset/` directory.
2. **Training**: Train the TACSSL model by running the main training script:
   ```bash
   python train/TACSSL.py
   ```
3. **Fine-tuning**: Use `finetune.py` for downstream tasks or fine-tuning.
4. **Visualization**: Plot training metrics and compare results using `plot_lambda_metrics.py`.

