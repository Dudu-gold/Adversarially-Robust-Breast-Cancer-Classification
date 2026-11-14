# ============================================================
# ADVERSARIALLY ROBUST BREAST CANCER CLASSIFICATION
# Google Colab Notebook
# Dataset: BreakHis (400x magnification)
# Model: ResNet-50 + PGD Adversarial Training
# ============================================================

# ─────────────────────────────────────────────
# STEP 1: INSTALL REQUIRED LIBRARIES
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# STEP 2: MOUNT GOOGLE DRIVE
# ─────────────────────────────────────────────


# ─────────────────────────────────────────────
# STEP 3: IMPORTS
# ─────────────────────────────────────────────
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# STEP 4: SET SEED FOR REPRODUCIBILITY
# ─────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# ─────────────────────────────────────────────
# STEP 5: CONFIG
# ─────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR    = "/content/drive/MyDrive/BreakHis/400X"  # <-- UPDATE THIS PATH
MODEL_DIR   = "/content/drive/MyDrive/breast_cancer_models"
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_BASE = 30
EPOCHS_ADV  = 50
LR          = 1e-4
LR_ADV      = 5e-5
WEIGHT_DECAY= 1e-5
EPS         = 8/255       # adversarial perturbation budget
ALPHA       = EPS / 10    # PGD step size
PGD_STEPS   = 40          # PGD iterations

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# STEP 6: DATA TRANSFORMS
# ─────────────────────────────────────────────
# ImageNet mean and std for ResNet-50
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ─────────────────────────────────────────────
# STEP 7: LOAD & SPLIT DATASET
# ─────────────────────────────────────────────
# BreakHis folder structure expected:
# 400X/
#   benign/   (images)
#   malignant/(images)

full_dataset = ImageFolder(root=DATA_DIR, transform=test_transform)
print(f"Classes: {full_dataset.classes}")
print(f"Total images: {len(full_dataset)}")

# Stratified 70/15/15 split
indices = list(range(len(full_dataset)))
labels  = [full_dataset.targets[i] for i in indices]

train_idx, temp_idx, _, temp_labels = train_test_split(
    indices, labels, test_size=0.30, stratify=labels, random_state=42)

val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=temp_labels, random_state=42)

print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# Apply train transforms to training subset
from torch.utils.data import Subset
train_subset = Subset(full_dataset, train_idx)
val_subset   = Subset(full_dataset, val_idx)
test_subset  = Subset(full_dataset, test_idx)

# Override train transform
train_dataset = ImageFolder(root=DATA_DIR, transform=train_transform)
train_subset  = Subset(train_dataset, train_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Class weights for imbalanced dataset
class_counts = np.bincount(labels)
class_weights = torch.tensor(
    [1.0 / class_counts[0], 1.0 / class_counts[1]], dtype=torch.float).to(DEVICE)
print(f"Class weights: {class_weights}")

# ─────────────────────────────────────────────
# STEP 8: BUILD RESNET-50 MODEL
# ─────────────────────────────────────────────
def build_resnet50(freeze_early=True):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    if freeze_early:
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
    # Replace final FC layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)

# ─────────────────────────────────────────────
# STEP 9: TRAINING HELPERS
# ─────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in tqdm(loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs   = torch.softmax(outputs, dim=1)[:, 1]
            preds   = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='macro')
    auc  = roc_auc_score(all_labels, all_probs)
    cm   = confusion_matrix(all_labels, all_preds)
    return acc, f1, auc, cm, all_preds, all_labels


# ─────────────────────────────────────────────
# STEP 10: TRAIN BASELINE MODEL
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("TRAINING BASELINE RESNET-50")
print("="*50)

baseline_model = build_resnet50(freeze_early=True)
criterion      = nn.CrossEntropyLoss(weight=class_weights)
optimizer_base = optim.Adam(
    filter(lambda p: p.requires_grad, baseline_model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_base = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_base, patience=3, verbose=True)

best_val_acc = 0
train_losses, val_accs = [], []

for epoch in range(EPOCHS_BASE):
    train_loss, train_acc = train_epoch(
        baseline_model, train_loader, criterion, optimizer_base)
    val_acc, val_f1, val_auc, _, _, _ = evaluate(baseline_model, val_loader)
    scheduler_base.step(1 - val_acc)

    train_losses.append(train_loss)
    val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS_BASE}] "
          f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
          f"Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(baseline_model.state_dict(),
                   os.path.join(MODEL_DIR, "baseline_resnet50.pth"))
        print(f"  --> Best model saved (Val Acc: {val_acc:.4f})")

print(f"\nBaseline training complete. Best Val Acc: {best_val_acc:.4f}")

# ─────────────────────────────────────────────
# STEP 11: PGD ADVERSARIAL ATTACK (inner loop)
# ─────────────────────────────────────────────
def pgd_attack(model, images, labels, eps=EPS, alpha=ALPHA, steps=PGD_STEPS):
    """Generate PGD adversarial examples."""
    model.eval()
    images = images.clone().detach().to(DEVICE)
    labels = labels.clone().detach().to(DEVICE)
    # Random initialisation within eps ball
    delta = torch.empty_like(images).uniform_(-eps, eps).to(DEVICE)
    adv_images = (images + delta).clamp(0, 1).detach()

    for _ in range(steps):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        grad = adv_images.grad.sign()
        adv_images = adv_images.detach() + alpha * grad
        delta = torch.clamp(adv_images - images, -eps, eps)
        adv_images = torch.clamp(images + delta, 0, 1).detach()

    return adv_images


# ─────────────────────────────────────────────
# STEP 12: EVALUATE BASELINE UNDER ATTACKS
# ─────────────────────────────────────────────
def fgsm_attack(model, images, labels, eps):
    images = images.clone().detach().requires_grad_(True).to(DEVICE)
    labels = labels.to(DEVICE)
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    return torch.clamp(images + eps * images.grad.sign(), 0, 1).detach()


def evaluate_under_attack(model, loader, attack_fn, eps):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        adv_imgs = attack_fn(model, imgs, labels, eps)
        with torch.no_grad():
            outputs = model(adv_imgs)
            preds   = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)


print("\n" + "="*50)
print("ADVERSARIAL ATTACK EVALUATION ON BASELINE")
print("="*50)

# Load best baseline
baseline_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "baseline_resnet50.pth")))

epsilons = [0.01, 0.03, 0.05]
print(f"\n{'Attack':<12} {'ε=0.01':>10} {'ε=0.03':>10} {'ε=0.05':>10}")
print("-" * 45)

# FGSM
fgsm_results = [evaluate_under_attack(
    baseline_model, test_loader, fgsm_attack, e) for e in epsilons]
print(f"{'FGSM':<12} {fgsm_results[0]:>10.4f} {fgsm_results[1]:>10.4f} {fgsm_results[2]:>10.4f}")

# PGD
pgd_results = [evaluate_under_attack(
    baseline_model, test_loader, pgd_attack, e) for e in epsilons]
print(f"{'PGD':<12} {pgd_results[0]:>10.4f} {pgd_results[1]:>10.4f} {pgd_results[2]:>10.4f}")

# ─────────────────────────────────────────────
# STEP 13: PGD ADVERSARIAL TRAINING
# ─────────────────────────────────────────────
print("\n" + "="*50)
print("PGD ADVERSARIAL TRAINING — RESNET-50")
print("="*50)

robust_model   = build_resnet50(freeze_early=False)  # Unfreeze all layers
criterion_adv  = nn.CrossEntropyLoss(weight=class_weights)
optimizer_adv  = optim.Adam(robust_model.parameters(),
                             lr=LR_ADV, weight_decay=WEIGHT_DECAY)
scheduler_adv  = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_adv, patience=3, verbose=True)

best_robust_acc = 0
adv_train_losses, adv_val_accs = [], []

for epoch in range(EPOCHS_ADV):
    robust_model.train()
    total_loss, correct, total = 0, 0, 0

    for imgs, labels in tqdm(train_loader, leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        # Generate PGD adversarial examples
        adv_imgs = pgd_attack(robust_model, imgs, labels)
        robust_model.train()
        optimizer_adv.zero_grad()
        outputs  = robust_model(adv_imgs)
        loss     = criterion_adv(outputs, labels)
        loss.backward()
        optimizer_adv.step()
        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += labels.size(0)

    train_acc = correct / total
    val_acc, val_f1, val_auc, _, _, _ = evaluate(robust_model, val_loader)
    scheduler_adv.step(1 - val_acc)
    adv_train_losses.append(total_loss / len(train_loader))
    adv_val_accs.append(val_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS_ADV}] "
          f"Adv Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
          f"Val AUC: {val_auc:.4f}")

    if val_acc > best_robust_acc:
        best_robust_acc = val_acc
        torch.save(robust_model.state_dict(),
                   os.path.join(MODEL_DIR, "robust_resnet50.pth"))
        print(f"  --> Best robust model saved (Val Acc: {val_acc:.4f})")

print(f"\nAdversarial training complete. Best Val Acc: {best_robust_acc:.4f}")

# ─────────────────────────────────────────────
# STEP 14: FINAL EVALUATION — BASELINE vs ROBUST
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("FINAL COMPARATIVE EVALUATION")
print("="*60)

# Load best models
baseline_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "baseline_resnet50.pth")))
robust_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "robust_resnet50.pth")))

# Clean evaluation
b_acc, b_f1, b_auc, b_cm, _, _ = evaluate(baseline_model, test_loader)
r_acc, r_f1, r_auc, r_cm, _, _ = evaluate(robust_model,   test_loader)

print(f"\n{'Metric':<25} {'Baseline':>12} {'Robust':>12}")
print("-" * 50)
print(f"{'Clean Accuracy':<25} {b_acc:>12.4f} {r_acc:>12.4f}")
print(f"{'F1-Score (Macro)':<25} {b_f1:>12.4f} {r_f1:>12.4f}")
print(f"{'AUC-ROC':<25} {b_auc:>12.4f} {r_auc:>12.4f}")

# Adversarial evaluation at eps=0.03
b_pgd = evaluate_under_attack(baseline_model, test_loader, pgd_attack, 0.03)
r_pgd = evaluate_under_attack(robust_model,   test_loader, pgd_attack, 0.03)
b_fgsm= evaluate_under_attack(baseline_model, test_loader, fgsm_attack, 0.03)
r_fgsm= evaluate_under_attack(robust_model,   test_loader, fgsm_attack, 0.03)

print(f"{'Robust Acc (FGSM ε=0.03)':<25} {b_fgsm:>12.4f} {r_fgsm:>12.4f}")
print(f"{'Robust Acc (PGD ε=0.03)':<25} {b_pgd:>12.4f} {r_pgd:>12.4f}")
print(f"{'Accuracy Drop (PGD)':<25} {b_acc-b_pgd:>12.4f} {r_acc-r_pgd:>12.4f}")

# ─────────────────────────────────────────────
# STEP 15: CONFUSION MATRIX PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_names = ["Benign", "Malignant"]

for ax, cm, title in zip(axes,
                          [b_cm, r_cm],
                          ["Baseline ResNet-50", "Adversarially Trained ResNet-50"]):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_names, yticklabels=labels_names, ax=ax)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrices.png"), dpi=150)
plt.show()
print("Confusion matrices saved.")

# ─────────────────────────────────────────────
# STEP 16: GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights     = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam         = (weights * self.activations).sum(dim=1, keepdim=True)
        cam         = torch.relu(cam)
        cam         = cam.squeeze().cpu().numpy()
        cam         = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def overlay_gradcam(image_tensor, cam, alpha=0.4):
    """Overlay GradCAM heatmap on original image."""
    img = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img = img * np.array(STD) + np.array(MEAN)
    img = np.clip(img, 0, 1)
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = alpha * heatmap + (1 - alpha) * img
    return np.clip(overlay, 0, 1)


# Generate Grad-CAM for a sample test image
gradcam    = GradCAM(robust_model, robust_model.layer4[-1])
sample_img, sample_label = next(iter(test_loader))
sample_img   = sample_img[:1].to(DEVICE)
cam          = gradcam.generate(sample_img)
overlay      = overlay_gradcam(sample_img, cam)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
orig = sample_img.squeeze().permute(1,2,0).cpu().numpy()
orig = np.clip(orig * np.array(STD) + np.array(MEAN), 0, 1)
axes[0].imshow(orig);        axes[0].set_title("Original Image");   axes[0].axis('off')
axes[1].imshow(overlay);     axes[1].set_title("Grad-CAM Overlay"); axes[1].axis('off')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "gradcam_sample.png"), dpi=150)
plt.show()
print("Grad-CAM saved.")

print("\n✅ ALL DONE — Models saved to:", MODEL_DIR)
