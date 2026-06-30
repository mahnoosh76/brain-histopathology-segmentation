# train.py

from matplotlib.colors import BoundaryNorm, ListedColormap
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import cv2

COLOR_MAP = {
    0: (0,   0,   0),
    1: (255, 0,   0),
    2: (0,   255, 0),
    3: (0,   0,   255),
    4: (255, 255, 0),
    5: (255, 0,   255),
}


os.makedirs("results", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)


def dice_loss(pred, target, smooth=1.):
    """
    pred: output from model => shape [B, C, H, W]
    target: ground truth => shape [B, H, W]
    """
    pred = torch.softmax(pred, dim=1)  # apply softmax across channels
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.shape[1])  # [B, H, W, C]
    target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()  # [B, C, H, W]

    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))

    dice = (2. * intersection + smooth) / (union + smooth)
    loss = 1 - dice.mean()

    return loss


def compute_class_weights(masks, num_classes=2):
    """Compute class weights from a list or array of masks."""
    all_labels = np.concatenate([mask.flatten() for mask in masks])
    counts = Counter(all_labels.tolist())

    total_pixels = sum(counts.values())
    weights = []

    for c in range(num_classes):
        pixel_count = counts.get(c, 1)  # avoid division by zero
        weight = total_pixels / (num_classes * pixel_count)
        weights.append(weight)

    normed = np.array(weights) / np.sum(weights)  # optional: normalize
    print(f"Class weights: {normed}")
    return torch.tensor(normed, dtype=torch.float32)


def colorize_mask(mask):
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_val, color in COLOR_MAP.items():
        color_img[mask == class_val] = color
    return color_img

def show_predictions(model, dataloader, device, epoch=0, num=1):
    model.eval()
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            if i >= num:
                break
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            imgs = images.cpu().permute(0, 2, 3, 1).numpy() * 255
            masks_np = masks.cpu().numpy()

            for j in range(images.shape[0]):
                img = imgs[j].astype(np.uint8)
                gt_mask = masks_np[j]
                pred_mask = preds[j]

                # Create figure with 3 panels
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))

                # 1) Raw Input
                axs[0].imshow(img)
                axs[0].set_title("Input Image")

                # Shared function to overlay mask
                def overlay(ax, base_img, mask, title):
                    ax.imshow(base_img)
                    ax.imshow(mask, cmap=ListedColormap([(0,0,0,0), (1,0,0,0.5)]),
                              norm=BoundaryNorm([-0.5, 0.5, 1.5], ncolors=2),
                              interpolation="nearest")
                    ax.set_title(title)
                    ax.axis("off")

                # 2) Ground Truth Overlay
                overlay(axs[1], img, gt_mask, "GT Overlay")

                # 3) Prediction Overlay
                overlay(axs[2], img, pred_mask, "Pred Overlay")

                plt.tight_layout()
                save_path = f"results/epoch_{epoch+1}_sample_{j}.png"
                plt.savefig(save_path)
                plt.close(fig)

            break



def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-4, y_train_masks=None, num_classes=2, task="segmentation"):
    class_weights = compute_class_weights(y_train_masks, num_classes=num_classes).to(device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights) if y_train_masks is not None \
            else nn.CrossEntropyLoss()
    
    def loss_fn(output, target):
        if y_train_masks is not None:
            return 0.5 * ce_loss(output, target) + 0.5 * dice_loss(output, target)
        else:
            return ce_loss(output, target)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    history = {'train_acc': [], 'val_acc': [], 'train_loss': [], 'val_loss': []}

    best_val = float('inf')
    patience = 10
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == masks).sum().item()
            train_total += torch.numel(masks)

        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        history['train_acc'].append(train_acc)
        history['train_loss'].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                preds = torch.argmax(outputs, dim=1)

                val_loss += loss.item()
                val_correct += (preds == masks).sum().item()
                val_total += torch.numel(masks)

        val_acc = val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(avg_val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if (epoch + 1) % 10 == 0:
            show_predictions(model, val_loader, device, epoch, num=1)

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    torch.save(model.state_dict(), "checkpoints/final_model.pth")
    print("Training complete. Model saved to checkpoints/final_model.pth")

    return history
