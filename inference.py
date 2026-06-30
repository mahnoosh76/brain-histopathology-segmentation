# inference.py

import torch
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from model_selector import get_model
from train import colorize_mask
from matplotlib.colors import ListedColormap, BoundaryNorm


def load_trained_model(checkpoint_path, num_classes, device):
    model = get_model("unet", task="segmentation", out_channels=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
    return img_tensor



def infer_and_visualize(
    model,
    image_paths,
    mask_paths=None,
    size=(256, 256),
    device='cpu',
    limit=20
):
    """
    Load images, run inference, and overlay both ground-truth and predicted masks on the input image.

    Args:
        model: trained PyTorch model
        image_paths: list of image file paths
        mask_paths: optional list of corresponding mask file paths
        size: tuple (H, W) for resizing images and masks
        device: torch device for model and inputs
        limit: max number of images to process
    """
    # Set up discrete colormap for binary mask overlay
    colors = [
        (0.0, 0.0, 0.0, 0.0),  # class 0 - transparent
        (1.0, 0.0, 0.0, 0.5),  # class 1 - red
        (0.0, 1.0, 0.0, 0.5),  # class 2 - green
        (0.0, 0.0, 1.0, 0.5),  # class 3 - blue
        (1.0, 1.0, 0.0, 0.5),  # class 4 - yellow
        (1.0, 0.0, 1.0, 0.5),  # class 5 - magenta
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=list(range(len(colors)+1)), ncolors=cmap.N)


    model.eval()
    for i, path in enumerate(image_paths[:limit]):
        # Read and resize image
        img_bgr = cv2.imread(path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, size)

        # Preprocess and infer
        input_tensor = preprocess_image(path, size).to(device)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Load and resize ground truth mask if available
        gt_mask = None
        if mask_paths is not None:
            mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

        # Plot original, GT overlay, prediction overlay
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Original image
        axes[0].imshow(img_resized)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # # Ground truth overlay
        # axes[1].imshow(img_resized)
        # if gt_mask is not None:
        #     axes[1].imshow(gt_mask, cmap=cmap, norm=norm, interpolation='nearest')
        # axes[1].set_title('Ground Truth Overlay')
        # axes[1].axis('off')

        # Prediction overlay
        axes[1].imshow(img_resized)
        axes[1].imshow(pred, cmap=cmap, norm=norm, interpolation='nearest')
        axes[1].set_title('Prediction Overlay')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), "predict_results", f"result_{i}.png"))
        plt.show()



# Example usage (from main script):
# model = load_trained_model("checkpoints/final_model.pth", 3, DEVICE)
# infer_and_visualize(model, test_paths, size=(256, 256), device=DEVICE, limit=10)
