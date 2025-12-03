import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import torch
from torchvision import transforms
import numpy as np
import os

from glcm_shufflenet import FusionModel, compute_glcm_features


# ===============================
# LOAD MODEL
# ===============================
MODEL_PATH = "best_model.pth"  # sesuaikan jika lokasinya berbeda

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
class_to_idx = checkpoint["class_to_idx"]
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Dummy untuk hitung dimensi GLCM
dummy_img = Image.new("RGB", (128, 128), (128, 128, 128))
glcm_dim = compute_glcm_features(dummy_img, distances=[1,2,3]).shape[0]

model = FusionModel(glcm_dim=glcm_dim, num_classes=len(class_to_idx), pretrained=False)
model.load_state_dict(checkpoint["model_state"])
model.eval()


# ===============================
# TRANSFORM Gambar
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])


# ===============================
# FUNGSI PREDIKSI
# ===============================
def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")

    # glcm
    glcm_input = img.resize((128,128))
    glcm_feat = compute_glcm_features(glcm_input, distances=[1,2,3])
    glcm_feat = torch.from_numpy(glcm_feat).unsqueeze(0)

    # transform gambar
    img_tensor = transform(img).unsqueeze(0)

    # forward
    with torch.no_grad():
        out = model(img_tensor, glcm_feat)
        pred_idx = out.argmax(dim=1).item()

    return idx_to_class[pred_idx]


# ===============================
# GUI TKINTER
# ===============================
root = tk.Tk()
root.title("Coffee Bean Quality Classifier")
root.geometry("600x700")

label_title = Label(root, text="Deteksi Kualitas Biji Kopi", font=("Arial", 18))
label_title.pack(pady=10)

image_label = Label(root)
image_label.pack(pady=10)

result_label = Label(root, text="", font=("Arial", 16))
result_label.pack(pady=10)


# ===============================
# FUNGSI BUTTON: PILIH FILE
# ===============================
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
    )

    if not file_path:
        return

    # Tampilkan gambar di GUI
    img = Image.open(file_path)
    img_display = img.resize((300, 300))
    img_tk = ImageTk.PhotoImage(img_display)
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    # Prediksi
    prediction = predict_image(file_path)
    result_label.configure(text=f"Hasil Prediksi: {prediction}")


# ===============================
# Tombol pilih gambar
# ===============================
btn_select = tk.Button(root, text="Pilih Gambar", font=("Arial", 14), command=open_file)
btn_select.pack(pady=20)


root.mainloop()