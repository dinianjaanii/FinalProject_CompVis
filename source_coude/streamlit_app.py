# streamlit_app.py
import streamlit as st
from PIL import Image
import torch
import numpy as np
import os
import pandas as pd
from torchvision import transforms
from glcm_shufflenet import FusionModel, compute_glcm_features
from gradcam import make_gradcam  # helper script provided below

MODEL_PATH = "best_model.pth"
GLCM_DISTANCES = [1,2,3]
IMG_SIZE = 224

@st.cache_resource
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    class_to_idx = checkpoint["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    dummy = Image.new("RGB",(128,128),(128,128,128))
    glcm_dim = compute_glcm_features(dummy, distances=GLCM_DISTANCES).shape[0]
    model = FusionModel(glcm_dim=glcm_dim, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, idx_to_class

model, idx_to_class = load_model()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

st.title("Coffee Bean Quality — Streamlit Demo")
st.write("Upload satu gambar atau pilih folder lokal (server) untuk batch inference.")

col1, col2 = st.columns(2)

with col1:
    uploaded = st.file_uploader("Upload gambar (jpg/png):", type=["jpg","jpeg","png","bmp"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Input", use_column_width=True)
        # predict
        glcm = compute_glcm_features(img.resize((128,128)), distances=GLCM_DISTANCES)
        glcm_t = torch.from_numpy(glcm).unsqueeze(0)
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(img_t, glcm_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(out.argmax(dim=1).item())
            pred_label = idx_to_class[pred_idx]
        st.markdown(f"**Prediction:** `{pred_label}`")
        # show probabilities
        df = pd.DataFrame({
            "class": [idx_to_class[i] for i in range(len(probs))],
            "probability": probs
        }).sort_values("probability", ascending=False)
        st.table(df.set_index("class"))

        # Grad-CAM toggle
        if st.checkbox("Show Grad-CAM heatmap"):
            heatmap = make_gradcam(img, model, pred_idx)
            st.image(heatmap, caption="Grad-CAM overlay", use_column_width=True)

with col2:
    st.header("Batch inference (server folder)")
    st.write("Provide a folder path on the machine running this app (e.g., E:/.../dataset/test/1_hitam)")
    folder = st.text_input("Folder path (optional):", value="")
    out_dir = st.text_input("Output CSV path (optional):", value="batch_results.csv")
    if st.button("Run batch inference") and folder:
        files = []
        for ext in (".jpg",".jpeg",".png",".bmp",".tiff"):
            for fn in os.listdir(folder):
                if fn.lower().endswith(ext):
                    files.append(os.path.join(folder,fn))
        if not files:
            st.error("No images found in folder.")
        else:
            results = []
            progress = st.progress(0)
            for i,fp in enumerate(files):
                img = Image.open(fp).convert("RGB")
                glcm = compute_glcm_features(img.resize((128,128)), distances=GLCM_DISTANCES)
                glcm_t = torch.from_numpy(glcm).unsqueeze(0)
                img_t = transform(img).unsqueeze(0)
                with torch.no_grad():
                    out = model(img_t, glcm_t)
                    probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                    pred_idx = int(out.argmax(dim=1).item())
                    pred_label = idx_to_class[pred_idx]
                results.append({"file":fp, "pred":pred_label, "score":float(probs[pred_idx])})
                progress.progress((i+1)/len(files))
            df = pd.DataFrame(results)
            df.to_csv(out_dir, index=False)
            st.success(f"Batch done — saved {len(files)} rows to {out_dir}")
            st.dataframe(df.head(200))