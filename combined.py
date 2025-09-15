import os, time
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FasterRCNN
from ultralytics import YOLO

import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

import streamlit as st

# XAI imports
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from captum.attr import IntegratedGradients, Occlusion

# =========================
# Streamlit Config
# =========================
st.set_page_config(page_title="AI Explainability Dashboard", layout="wide")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Helper: Explanation Text
# =========================
def explanation_text(method, class_name="the predicted class/defect"):
    texts = {
        "gradcam": f"**Grad-CAM** highlights the regions most responsible for predicting **{class_name}**. "
                   "Warm colors (red/yellow) = stronger influence.",
        "ig": f"**Integrated Gradients** attribute importance pixel-wise for **{class_name}**. "
              "Clear highlights show where the model is relying.",
        "occ": f"**Occlusion Sensitivity** checks which regions reduce confidence when hidden. "
               f"Important zones for **{class_name}** light up strongly.",
        "sal": f"**Saliency Maps** capture gradients to show which pixels most affect the prediction of **{class_name}**."
    }
    return texts[method]

# =========================
# YOLOv8 Classification Setup
# =========================
class ModelWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        out = self.m(x)
        return out[0] if isinstance(out, (tuple, list)) else out

# Match training preprocessing: resize + scale to [0,1], no ImageNet normalization
transform_yolo = T.Compose([
    T.Resize((300, 300)),
    T.ToTensor(),
])

@st.cache_resource
def load_yolo():
    model = YOLO("best.pt")   # ✅ correct path
    net = model.model
    net.eval()
    wrapped = ModelWrapper(net).to(DEVICE)
    return model, wrapped

yolo_model, yolo_wrapped = load_yolo()

# =========================
# FasterRCNN Setup
# =========================
CLASS_NAMES = [
    "welding_line", "water_spot", "waist_folding", "silk_spot",
    "punching_hole", "rolled_pit", "oil_spot", "inclusion",
    "crescent_gap", "crease"
]

LABEL_MAP = {
    '1_chongkong': 'punching_hole',
    '2_hanfeng': 'welding_line',
    '3_yueyawan': 'crescent_gap',
    '4_shuiban': 'water_spot',
    '5_youban': 'oil_spot',
    '6_siban': 'silk_spot',
    '7_yiwu': 'inclusion',
    '8_yahen': 'rolled_pit',
    '9_zhehen': 'crease',
    '10_yaozhed': 'waist_folding'
}

LABELS_DIR = r"C:\Users\me1mna\Downloads\Defects location for metal surface\label\label"

torch.serialization.add_safe_globals([FasterRCNN])

@st.cache_resource
def load_fastercnn():
    checkpoint_path = "fasterrcnn_metal_defects_full50.pth"
    try:
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=len(CLASS_NAMES)+1)
        state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.warning(f"State_dict load failed ({e}). Trying full model load...")
        model = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model.to(DEVICE).eval()
    return model

faster_model = load_fastercnn()

# =========================
# FasterRCNN Prediction Utils
# =========================
def parse_xml_file(xml_path):
    root = ET.parse(xml_path).getroot()
    objs = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name")
        name = LABEL_MAP.get(raw_name, raw_name)
        if name not in CLASS_NAMES: continue
        bb = obj.find("bndbox")
        xmin = int(bb.findtext("xmin"))
        ymin = int(bb.findtext("ymin"))
        xmax = int(bb.findtext("xmax"))
        ymax = int(bb.findtext("ymax"))
        objs.append((name, (xmin, ymin, xmax, ymax)))
    return objs

def get_gt_for_image(img_path):
    xml_name = os.path.splitext(os.path.basename(img_path))[0] + ".xml"
    xml_path = os.path.join(LABELS_DIR, xml_name)
    if os.path.exists(xml_path):
        return parse_xml_file(xml_path)
    return []

transform_det = T.Compose([T.ToTensor()])

def predict_and_plot(image_pil, model, score_thresh=0.5):
    image_tensor = transform_det(image_pil).unsqueeze(0).to(DEVICE)
    outputs = model(image_tensor)

    boxes = outputs[0]['boxes'].detach().cpu().numpy()
    labels = outputs[0]['labels'].detach().cpu().numpy()
    scores = outputs[0]['scores'].detach().cpu().numpy()

    # --- Ground Truth ---
    img_gt = image_pil.copy()
    draw_gt = ImageDraw.Draw(img_gt)
    gt_boxes = get_gt_for_image(image_pil.filename)
    for name, (xmin, ymin, xmax, ymax) in gt_boxes:
        draw_gt.rectangle([xmin, ymin, xmax, ymax], outline="green", width=4)
        draw_gt.text((xmin, max(ymin-25, 0)), f"GT: {name}", fill="green")

    # --- Predictions ---
    img_pred = image_pil.copy()
    draw_pred = ImageDraw.Draw(img_pred)
    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh: continue
        x1, y1, x2, y2 = box
        pred_name = CLASS_NAMES[label-1]
        draw_pred.rectangle([x1, y1, x2, y2], outline="red", width=4)
        draw_pred.text((x1, max(y1-25, 0)), f"{pred_name} {score:.2f}", fill="red")

    return img_gt, img_pred, image_tensor, scores

# =========================
# Streamlit App
# =========================
st.title("🔍 AI Explainability Dashboard")
st.markdown("This dashboard provides **visual + textual explanations** for YOLOv8 (classification) and FasterRCNN (detection).")

tab1, tab2 = st.tabs(["YOLOv8 Classification", "FasterRCNN Detection"])

# -------------------------
# TAB 1: YOLOv8 Classification
# -------------------------
with tab1:
    st.header("YOLOv8 Classification + XAI")

    uploaded_file = st.file_uploader("Upload an image for YOLOv8 classification", type=["jpg","png","jpeg"], key="yolo")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        rgb = np.array(img)

        inp = transform_yolo(img).unsqueeze(0).to(DEVICE)
        inp.requires_grad_(True)

        scores = yolo_wrapped(inp)
        pred = int(scores.argmax(dim=1).item())
        class_name = yolo_model.names[pred]
        st.markdown(f"### Prediction: **{class_name}**")

        # ---- Explanations horizontally ----
        cols = st.columns(5)

        # Original
        with cols[0]:
            st.image(rgb, caption=f"Original\nPrediction: {class_name}", width=250)

        # Grad-CAM
        def last_conv(module: nn.Module):
            last = None
            for m in module.modules():
                if isinstance(m, nn.Conv2d): last = m
            return last
        target_layer = last_conv(yolo_wrapped)
        with GradCAM(model=yolo_wrapped, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(pred)])[0]
        gradcam_overlay = show_cam_on_image(rgb.astype(np.float32)/255.0, grayscale_cam, use_rgb=True)
        with cols[1]:
            st.image(gradcam_overlay, caption="Grad-CAM", width=250)
            with st.expander("Explanation"):
                st.markdown(explanation_text("gradcam", class_name))

        # Integrated Gradients
        ig = IntegratedGradients(yolo_wrapped)
        baseline = torch.zeros_like(inp)
        attributions_ig = ig.attribute(inp, baselines=baseline, target=pred, n_steps=25, internal_batch_size=4)
        attr = attributions_ig[0].detach().cpu().numpy().transpose(1,2,0)
        attr = np.mean(np.abs(attr), axis=-1)
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        ig_overlay = (0.5*cv2.applyColorMap(np.uint8(attr*255), cv2.COLORMAP_JET) + 0.5*bgr).astype(np.uint8)
        with cols[2]:
            st.image(cv2.cvtColor(ig_overlay, cv2.COLOR_BGR2RGB), caption="Integrated Gradients", width=250)
            with st.expander("Explanation"):
                st.markdown(explanation_text("ig", class_name))

        # Occlusion
        occ = Occlusion(yolo_wrapped)
        attr_occ = occ.attribute(inp, target=pred, strides=(3,20,20), sliding_window_shapes=(3,40,40), baselines=0)
        hm = attr_occ[0].detach().cpu().numpy().sum(axis=0)
        hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        occ_overlay = (0.5*cv2.applyColorMap(np.uint8(hm*255), cv2.COLORMAP_JET) + 0.5*bgr).astype(np.uint8)
        with cols[3]:
            st.image(cv2.cvtColor(occ_overlay, cv2.COLOR_BGR2RGB), caption="Occlusion", width=250)
            with st.expander("Explanation"):
                st.markdown(explanation_text("occ", class_name))

        # Saliency
        scores[:, pred].backward(retain_graph=True)
        sal = inp.grad[0].detach().cpu().numpy().transpose(1,2,0)
        sal = np.max(np.abs(sal), axis=-1)
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
        sal_overlay = (0.5*cv2.applyColorMap(np.uint8(sal*255), cv2.COLORMAP_JET) + 0.5*bgr).astype(np.uint8)
        with cols[4]:
            st.image(cv2.cvtColor(sal_overlay, cv2.COLOR_BGR2RGB), caption="Saliency Map", width=250)
            with st.expander("Explanation"):
                st.markdown(explanation_text("sal", class_name))

# -------------------------
# TAB 2: FasterRCNN Detection
# -------------------------
with tab2:
    st.header("FasterRCNN Detection + XAI")

    uploaded_file = st.file_uploader("Upload an image for FasterRCNN detection", type=["jpg","png","jpeg"], key="faster")

    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_pil.filename = uploaded_file.name

        img_gt, img_pred, image_tensor, scores = predict_and_plot(image_pil, faster_model)
        target_idx = int(np.argmax(scores))

        st.markdown("### Ground Truth vs Predictions")
        cols = st.columns(2)
        with cols[0]:
            st.image(img_gt, caption="Ground Truth", width=400)
        with cols[1]:
            st.image(img_pred, caption="Predictions", width=400)

        # Grad-CAM
        gradients, activations = [], []
        def forward_hook(module, inp, out): activations.append(out)
        def backward_hook(module, grad_in, grad_out): gradients.append(grad_out[0])

        target_layer = faster_model.backbone.body.layer4
        fh = target_layer.register_forward_hook(forward_hook)
        bh = target_layer.register_full_backward_hook(backward_hook)

        output = faster_model(image_tensor)
        score = output[0]['scores'][target_idx]
        faster_model.zero_grad()
        score.backward()

        grads = gradients[0].detach()
        acts = activations[0].detach()
        weights = grads.mean(dim=[2,3], keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-6)
        fh.remove(); bh.remove()

        cam_resized = cv2.resize(cam.cpu().numpy(), image_pil.size)
        cols = st.columns(2)
        with cols[0]:
            fig, ax = plt.subplots()
            ax.imshow(image_pil)
            ax.imshow(cam_resized, cmap='jet', alpha=0.5)
            ax.axis("off")
            st.pyplot(fig)
            with st.expander("Explanation"):
                st.markdown(explanation_text("gradcam", "detected defect"))

        # Saliency
        image_tensor.requires_grad_()
        output = faster_model(image_tensor)
        score = output[0]['scores'][target_idx]
        score.backward()
        sal = image_tensor.grad.data.abs().squeeze().max(dim=0)[0]
        sal = (sal - sal.min()) / (sal.max() + 1e-6)
        with cols[1]:
            fig, ax = plt.subplots()
            ax.imshow(sal.cpu().numpy(), cmap='hot')
            ax.axis("off")
            st.pyplot(fig)
            with st.expander("Explanation"):
                st.markdown(explanation_text("sal", "detected defect"))
