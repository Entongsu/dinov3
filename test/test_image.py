import pickle
import os
import urllib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
from scipy import signal

# --- DINOv3 setup ---
REPO_DIR = "."  # your local repo (with hubconf.py)
CKPT_PATH = "ckpt/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

model = torch.hub.load(repo_or_dir=REPO_DIR,
                       model="dinov3_vits16",
                       source="local",
                       weights=CKPT_PATH).cuda().eval()

PATCH_SIZE = 16
IMAGE_SIZE = 768

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
image_uri = "https://dl.fbaipublicfiles.com/dinov3/notebooks/pca/test_image.jpg"


# --- Utilities ---
def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def resize_transform(mask_image: Image,
                     image_size: int = IMAGE_SIZE,
                     patch_size: int = PATCH_SIZE) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(
        TF.resize(mask_image,
                  (h_patches * patch_size, w_patches * patch_size)))


# --- Load and preprocess image ---
image = load_image_from_url(image_uri)
import cv2

img_bgr = cv2.imread("/home/ensu/Downloads/download.jpeg")  # BGR order
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
image = Image.fromarray(img_rgb)
image_resized = resize_transform(image)
image_resized_norm = TF.normalize(image_resized,
                                  mean=IMAGENET_MEAN,
                                  std=IMAGENET_STD)

# --- Extract patch features ---
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}

MODEL_NAME = "dinov3_vits16"
n_layers = MODEL_TO_NUM_LAYERS[MODEL_NAME]

with torch.inference_mode():
    feats = model.get_intermediate_layers(
        image_resized_norm.unsqueeze(0).cuda(),
        n=range(n_layers),
        reshape=True,
        norm=True)
    # last layer features
    x = feats[-1].squeeze().detach().cpu()
    dim = x.shape[0]
    x = x.view(dim, -1).permute(1, 0)

h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]

# --- Visualization: feature norm ---
feat_norm = torch.norm(x, dim=1)
feat_norm = (feat_norm - feat_norm.min()) / (feat_norm.max() - feat_norm.min())
feat_map = feat_norm.reshape(h_patches, w_patches)

# --- Visualization: PCA projection ---
pca = PCA(n_components=3)
feat_pca = pca.fit_transform(x)
feat_pca = (feat_pca - feat_pca.min(0)) / (feat_pca.max(0) - feat_pca.min(0))
feat_pca = feat_pca.reshape(h_patches, w_patches, 3)

# --- Plot both ---
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(feat_map, cmap="inferno")
plt.title("Feature Magnitude")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(feat_pca)
plt.title("Feature PCA Map (RGB)")
plt.axis("off")

plt.tight_layout()
plt.show()
