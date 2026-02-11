# Image Inpainting with Masked Autoencoders (MAE)

This project implements **Masked Autoencoders (MAE)**, a scalable self-supervised learner for computer vision. By masking random patches of an input image and reconstructing the missing pixels, the model learns high-level visual representations that are effective for image inpainting and downstream tasks.

## 🚀 Overview

The core idea of this project is to prove that image inpainting can serve as a powerful pretext task for visual representation learning. The architecture utilizes a **Vision Transformer (ViT)** backbone, where an encoder processes only the visible patches and a lightweight decoder reconstructs the original image from the latent representation and mask tokens.

## 🛠️ Key Features

* **Patch Embedding:** Images are divided into non-overlapping patches (16x16) and projected into a linear embedding space.
* **High Masking Ratio:** Implements a default masking ratio of **75%**, forcing the model to learn holistic structures rather than just interpolating local pixels.
* **Asymmetric Encoder-Decoder:** * **Encoder:** A deep ViT that only "sees" unmasked patches, reducing computational overhead.
* **Decoder:** A lightweight Transformer that reconstructs pixels from the latent space and learned mask tokens.


* **Evaluation Metrics:** Includes standard metrics such as **MSE Loss**, **LPIPS** (Learned Perceptual Image Patch Similarity), and **FID** (Frechet Inception Distance) to assess reconstruction quality.

## 📊 Results


### 1. Image Inpainting Performance

The model learned to reconstruct images from the **mini-ImageNet** dataset by "filling in" missing pieces.

* **Visual Quality:** The model successfully captured general shapes and structures. However, it struggled with fine details, leading to some blurriness or "blocky" patterns.
* **Masking Ratios:** * **50% Masking:** Performed better because the model had more original data to work with (LPIPS: 0.5610 | FID: 198.33).
* **75% Masking:** More challenging for the model, but still achieved decent results (LPIPS: 0.5851 | FID: 224.78).


* **The "Why":** The high LPIPS/FID scores (meaning lower similarity to real images) were likely due to the short training time (20 epochs) and the smaller dataset size compared to the full ImageNet.

### 2. Downstream Classification Results

This is where the model really proved its worth. We used the "brain" (Encoder) of the inpainting model to see if it could recognize objects.

| Experiment | Method | Test Accuracy |
| --- | --- | --- |
| **Supervised Baseline** | Training from scratch | **24.41%** |
| **Linear Probing** | Using fixed pre-trained features | **29.67%** |
| **Fine-Tuning** | Adjusting pre-trained features | **54.50%** |

* **The Big Win:** The **Fine-Tuned** model was **2x more accurate** than the baseline.
* **Speed:** The pre-trained model reached 37% accuracy in just its first round of training—already beating the baseline's final score.
* **Key Insight:** Even without labels, the model "learned" what objects look like just by trying to fix broken images.

---

## 💻 Tech Stack

* **Language:** Python
* **Deep Learning Framework:** PyTorch
* **Visualization:** Matplotlib, PIL
* **Utilities:** `torchvision`, `lpips`, `torch-fidelity`, `tqdm`

## 📖 How to Use

### 1. Installation

Install the required dependencies:

```bash
pip install torch torchvision lpips torch-fidelity tqdm

```

### 2. Training

The model can be trained by running the `Main Project.ipynb` notebook. It includes data loading, resizing, and a 20-epoch training loop with a Cosine Annealing learning rate scheduler.

### 3. Inference/Inpainting

You can load the provided `best_mae_model.pth` to perform inpainting on your own images:

```python
# Load the model
model.load_state_dict(torch.load("best_mae_model.pth")["model_state_dict"])
model.eval()

# Run reconstruction
loss, pred, mask = model(images, mask_ratio=0.75)
reconstructed_img = model.unpatchify(pred)

```

---
