# Brain Histopathology Segmentation Using U-Net


This project uses a U-Net convolutional neural network for semantic segmentation of histopathological lesions in H&E-stained mouse brain images induced by arsenic poisoning.
## Dataset

### Data Collection 
The dataset was collected from histopathological slides of brain tissue from mice exposed to arsenic. The slides were obtained from the Pathology Laboratory of the Faculty of Veterinary Medicine, Ferdowsi University of Mashhad.

### Image Acquisition

A total of 350 H&E-stained images were captured from different areas of the histopathological slides using an Olympus BH2 light microscope at 40× magnification.The original images were RGB JPG files with a resolution of 1836 × 3264 pixels and 24-bit color depth.

### Annotation

The lesion areas were manually annotated using Digital Sreeni Annotator. Multi-class segmentation masks were created to label the different tissue and lesion types in the images. These masks were used as the ground-truth labels for training the U-Net model.

## Classes

The model segments the images into six classes:

* **0 — Background:** Areas without labeled tissue or lesions.
* **1 — Acute neural necrosis:** Areas showing acute neuronal cell death.
* **2 — Hyperemia:** Areas with increased blood flow and blood vessel congestion.
* **3 — Normal neuron:** Normal-appearing neuronal cells.
* **4 — Perineural edema:** Edema around neuronal cells.
* **5 — Perivascular edema:** Edema around blood vessels.
## Method / Model

I used a U-Net convolutional neural network for multi-class semantic segmentation of the histopathological images.
The images were resized to 256 × 256 pixels before training. I also applied data augmentation techniques, including horizontal and vertical flipping, rotation, noise, blur, and brightness adjustment.
The model takes an H&E-stained mouse brain image as input and generates a segmentation mask with six classes, including normal tissue and different types of histopathological lesions.
The model was implemented in PyTorch.
## Results

The model was evaluated separately for each class using Precision, Recall, F1-score, and AUC.

| Class            | Lesion type           |  Precision |     Recall |   F1-score |        AUC |
| ---------------- | --------------------- | ---------: | ---------: | ---------: | ---------: |
| 0                | Background            |     0.9797 |     0.9105 |     0.9439 |     0.9166 |
| 1                | Acute neural necrosis |     0.5585 |     0.8951 |     0.6878 |     0.9929 |
| 2                | Hyperemia             |     0.5685 |     0.0406 |     0.0758 |     0.4678 |
| 3                | Normal neuron         |     0.2302 |     0.8634 |     0.3635 |     0.9603 |
| 4                | Perineural edema      |     0.5007 |     0.5315 |     0.5156 |     0.9716 |
| 5                | Perivascular edema    |     0.4989 |     0.0427 |     0.0787 |     0.8314 |
| **Macro Avg**    |                       | **0.5561** | **0.5473** | **0.4442** | **0.8568** |
| **Weighted Avg** |                       | **0.9631** | **0.9502** | **0.9021** | **0.9232** |

**Mean IoU:** 0.3126
**Mean Dice:** 0.4084



## Features

- Multi-class semantic segmentation
- U-Net architecture implemented in PyTorch
- Histopathological image analysis
- Streamlit web application for inference

## Project Structure

- `train.py` – Model training
- `inference.py` – Prediction on new images
- `unet.py` – U-Net architecture
- `app.py` – Streamlit web application
- `model_selector.py` – Model loading utilities

## Author

Mahnoosh Parsa
