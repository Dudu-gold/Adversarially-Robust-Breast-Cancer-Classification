# ============================================================
# STREAMLIT WEB APPLICATION
# Adversarially Robust Breast Cancer Classification
# File: streamlit_app/app.py
# Run: streamlit run app.py
# ============================================================

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE   = 224
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
DEVICE     = torch.device("cpu")  # Use CPU for deployment
MODEL_PATH = "models/robust_resnet50.pth"  # Path to saved model
CLASSES    = ["Benign", "Malignant"]

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

# ─────────────────────────────────────────────
# GRAD-CAM
# ─────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        self.model.zero_grad()
        output[0, class_idx].backward()
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = torch.relu(cam).squeeze().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def generate_overlay(pil_image, cam):
    img = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
    cam_resized = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = 0.4 * heatmap + 0.6 * img
    return np.clip(overlay, 0, 1)


# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(pil_image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])
    return transform(pil_image).unsqueeze(0)


# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #ffffff; }
    .block-container { padding-top: 1.5rem; }
    .title-text {
        font-size: 2rem; font-weight: 700;
        color: #1F4E79; text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle-text {
        font-size: 1rem; color: #555555;
        text-align: center; margin-bottom: 1.5rem;
    }
    .result-benign {
        background-color: #EAFAF1; border-left: 6px solid #1E8449;
        padding: 1rem 1.5rem; border-radius: 6px;
        font-size: 1.1rem; font-weight: 600; color: #1E8449;
    }
    .result-malignant {
        background-color: #FDEDEC; border-left: 6px solid #C0392B;
        padding: 1rem 1.5rem; border-radius: 6px;
        font-size: 1.1rem; font-weight: 600; color: #C0392B;
    }
    .metric-box {
        background-color: #F2F4F7; border-radius: 6px;
        padding: 0.8rem 1rem; text-align: center;
    }
    .disclaimer {
        background-color: #FEF9E7; border: 1px solid #F9E79F;
        padding: 0.8rem 1rem; border-radius: 6px;
        font-size: 0.85rem; color: #7D6608; margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown('<div class="title-text">🔬 Adversarially Robust Breast Cancer Classifier</div>',
            unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">ResNet-50 with PGD Adversarial Training — BreakHis Dataset (400×)</div>',
            unsafe_allow_html=True)
st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Breast_cancer_cell.jpg/320px-Breast_cancer_cell.jpg",
             caption="Breast Tissue Histopathology", use_column_width=True)
    st.markdown("### About This System")
    st.markdown("""
This system classifies breast cancer histopathological images as **Benign** or **Malignant** using a ResNet-50 model trained with **PGD Adversarial Training** for robustness against adversarial attacks.

**Model:** ResNet-50 (ImageNet pretrained)

**Defense:** PGD Adversarial Training 

    """)
    st.divider()
    st.markdown("⚠️ **For research purposes only.**")

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
model    = load_model()
grad_cam = GradCAM(model, model.layer4[-1])

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📂 Upload Histopathological Image")
    uploaded_file = st.file_uploader(
        "Upload a breast tissue image (PNG or JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        pil_image = Image.open(uploaded_file).convert("RGB")
        st.image(pil_image, caption="Uploaded Image",
                 use_column_width=True)

with col2:
    if uploaded_file:
        st.subheader("🧠 Classification Result")

        with st.spinner("Analysing image..."):
            input_tensor = preprocess(pil_image)
            cam, pred_idx = grad_cam.generate(input_tensor)

            with torch.no_grad():
                output     = model(input_tensor)
                probs      = torch.softmax(output, dim=1)[0]
                confidence = probs[pred_idx].item() * 100
                prediction = CLASSES[pred_idx]

        # Result display
        if prediction == "Benign":
            st.markdown(
                f'<div class="result-benign">✅ Prediction: BENIGN<br>'
                f'Confidence: {confidence:.1f}%</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="result-malignant">⚠️ Prediction: MALIGNANT<br>'
                f'Confidence: {confidence:.1f}%</div>',
                unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar
        st.markdown("**Class Probabilities:**")
        st.progress(float(probs[0].item()), text=f"Benign: {probs[0].item()*100:.1f}%")
        st.progress(float(probs[1].item()), text=f"Malignant: {probs[1].item()*100:.1f}%")

        # Grad-CAM
        st.subheader("🗺️ Grad-CAM Explanation")
        st.caption("Highlighted regions show areas most influential in the model's prediction.")
        overlay = generate_overlay(pil_image, cam)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        orig = np.array(pil_image.resize((IMG_SIZE, IMG_SIZE))) / 255.0
        axes[0].imshow(orig);    axes[0].set_title("Original Image", fontsize=11)
        axes[0].axis('off')
        axes[1].imshow(overlay); axes[1].set_title("Grad-CAM Heatmap", fontsize=11)
        axes[1].axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        st.image(buf, use_column_width=True)
        plt.close()

        # Disclaimer
        st.markdown(
            '<div class="disclaimer">⚠️ <strong>Disclaimer:</strong> This system is a '
            'research decision-support tool and is not a substitute for formal clinical '
            'diagnosis by a qualified medical professional.</div>',
            unsafe_allow_html=True)

    else:
        st.info("👆 Please upload a histopathological image to begin classification.")
        st.markdown("""
**How to use:**
1. Upload a breast tissue histopathological image (PNG or JPEG)
2. The system will classify it as Benign or Malignant
3. A Grad-CAM heatmap will show which image regions influenced the prediction
4. Confidence scores will be displayed for both classes
        """)
