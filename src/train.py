"""
Training and Validation Pipeline for CNN Hybrid Calorie Estimator
Project: An Intelligent System to Estimate calorie content from meal photographs
Author: PALAGANI MONESHA (23ME1A54C7)
Supervisor: Dr. P. Sudhakar
Domain: Deep Learning & Computer Vision
"""

import os
import json
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from model import FoodCalorieHybridCNN, MultiTaskCalorieLoss
from dataset import (
    MealDataset, 
    load_nutrition_db, 
    FOOD_CLASSES, 
    CLASS_TO_IDX,
    create_sample_dataset
)


def train_model(
    manifest_path=None,
    epochs=15,
    batch_size=8,
    lr=1e-3,
    save_dir=None,
    device=None
):
    """
    Executes multi-task training of the CNN regression-classification hybrid.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if save_dir is None:
        save_dir = os.path.join(base_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    # If manifest not provided, check or generate sample benchmark dataset
    if manifest_path is None or not os.path.exists(manifest_path):
        sample_dir = os.path.join(base_dir, "sample_data")
        manifest_path = os.path.join(sample_dir, "dataset_manifest.json")
        if not os.path.exists(manifest_path):
            print("Benchmark dataset manifest not found. Generating sample dataset...")
            manifest_path = create_sample_dataset(sample_dir, samples_per_class=10)

    with open(manifest_path, "r") as f:
        data_records = json.load(f)

    print(f"Loaded {len(data_records)} meal samples from {manifest_path}")

    # Dataset split: 80% Train, 20% Validation
    full_dataset = MealDataset(data_records)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # Initialize Hybrid CNN model and Multi-Task criterion
    model = FoodCalorieHybridCNN(num_classes=len(FOOD_CLASSES), pretrained=True)
    model.to(device)

    criterion = MultiTaskCalorieLoss(cls_weight=1.0, cal_weight=1.5, macro_weight=0.8)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_mae = float('inf')
    best_model_path = os.path.join(save_dir, "best_hybrid_cnn.pth")
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_mae_kcal": [], "val_rmse_kcal": []
    }

    print("\n" + "="*60)
    print("STARTING CNN HYBRID MULTI-TASK TRAINING")
    print("="*60)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_cls = 0
        total_samples = 0

        for images, class_targets, macro_targets in train_loader:
            images = images.to(device)
            class_targets = class_targets.to(device)
            macro_targets = macro_targets.to(device)

            optimizer.zero_grad()
            logits, macro_preds = model(images)

            loss, loss_dict = criterion(logits, macro_preds, class_targets, macro_targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds_cls = torch.argmax(logits, dim=1)
            correct_cls += (preds_cls == class_targets).sum().item()
            total_samples += images.size(0)

        epoch_train_loss = running_loss / total_samples
        epoch_train_acc = (correct_cls / total_samples) * 100.0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct_cls = 0
        val_samples = 0
        cal_errors = []

        with torch.no_grad():
            for images, class_targets, macro_targets in val_loader:
                images = images.to(device)
                class_targets = class_targets.to(device)
                macro_targets = macro_targets.to(device)

                logits, macro_preds = model(images)
                loss, _ = criterion(logits, macro_preds, class_targets, macro_targets)

                val_loss += loss.item() * images.size(0)
                preds_cls = torch.argmax(logits, dim=1)
                val_correct_cls += (preds_cls == class_targets).sum().item()
                val_samples += images.size(0)

                # Calorie error calculation (pred vs true calories)
                pred_cals = macro_preds[:, 0].cpu().numpy()
                true_cals = macro_targets[:, 0].cpu().numpy()
                cal_errors.extend(np.abs(pred_cals - true_cals))

        epoch_val_loss = val_loss / val_samples
        epoch_val_acc = (val_correct_cls / val_samples) * 100.0
        val_mae = float(np.mean(cal_errors))
        val_rmse = float(np.sqrt(np.mean(np.array(cal_errors) ** 2)))

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(round(epoch_train_loss, 4))
        history["val_loss"].append(round(epoch_val_loss, 4))
        history["train_acc"].append(round(epoch_train_acc, 2))
        history["val_acc"].append(round(epoch_val_acc, 2))
        history["val_mae_kcal"].append(round(val_mae, 2))
        history["val_rmse_kcal"].append(round(val_rmse, 2))

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.3f} | Acc: {epoch_train_acc:.1f}% || "
              f"Val Loss: {epoch_val_loss:.3f} | Acc: {epoch_val_acc:.1f}% | "
              f"Calorie MAE: {val_mae:.1f} kcal | RMSE: {val_rmse:.1f} kcal")

        # Save Best Model Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae,
                'val_acc': epoch_val_acc,
                'classes': FOOD_CLASSES
            }, best_model_path)
            print(f"  --> Saved new best checkpoint (MAE: {val_mae:.2f} kcal)")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds. Best Val MAE: {best_val_mae:.2f} kcal")

    # Save metrics log for dashboard visualization
    history_path = os.path.join(save_dir, "metrics_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_model_path, history


if __name__ == "__main__":
    train_model(epochs=12, batch_size=8)
