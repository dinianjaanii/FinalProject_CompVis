import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import json

from glcm_shufflenet import FusionModel, compute_glcm_features


# ---------------------------
# Argumen
# ---------------------------
parser = argparse.ArgumentParser(description="Predict coffee bean defect")
parser.add_argument("--img_path", type=str, required=True, help="Path ke gambar input")
parser.add_argument("--model_path", type=str, default="best_model.pth", help="Path model tersimpan")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--glcm_distances", nargs='+', type=int, default=[1,2,3])
args = parser.parse_args()


# ---------------------------
# Load model & metadata
# ---------------------------
checkpoint = torch.load(args.model_path, map_location="cpu")

class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v:k for k,v in class_to_idx.items()}

# buat dummy untuk mengetahui dimensi GLCM
dummy_img = Image.new("RGB", (128,128), (128,128,128))
glcm_dim = compute_glcm_features(dummy_img, distances=args.glcm_distances).shape[0]

# model
model = FusionModel(glcm_dim=glcm_dim, num_classes=len(class_to_idx), pretrained=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()


# ---------------------------
# Transform gambar
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((args.img_size, args.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------------------------
# Inference Function
# ---------------------------
def predict_image(img_path):
    # load image
    img = Image.open(img_path).convert("RGB")

    # glcm feature (pakai ukuran kecil untuk stabil)
    glcm_input = img.resize((128,128))
    glcm_feats = compute_glcm_features(glcm_input, distances=args.glcm_distances)
    glcm_feats = torch.from_numpy(glcm_feats).unsqueeze(0)

    # transform ke tensor
    img_tensor = transform(img).unsqueeze(0)

    # forward
    with torch.no_grad():
        outputs = model(img_tensor, glcm_feats)
        pred_idx = outputs.argmax(dim=1).item()
        pred_class = idx_to_class[pred_idx]

    return pred_class


# ---------------------------
# Eksekusi
# ---------------------------
result = predict_image(args.img_path)
print(f"\n=== HASIL PREDIKSI ===")
print(f"Gambar: {args.img_path}")
print(f"Kelas prediksi: {result}")