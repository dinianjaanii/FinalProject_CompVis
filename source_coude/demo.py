import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

MODEL_PATH = 'model_fusifitur_kopi.pth'
CLASSES = ['Hitam', 'Pecah', 'Berjamur', 'Berlubang', 'Coklat', 'Muda']
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.cnn = models.shufflenet_v2_x1_0(weights=None)
        self.cnn.fc = nn.Identity()
        self.glcm_mlp = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Linear(5, 32),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1056, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_visual, x_texture):
        feat_visual = self.cnn(x_visual)
        feat_texture = self.glcm_mlp(x_texture)
        return self.classifier(torch.cat((feat_visual, feat_texture), dim=1))


def extract_glcm(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    feats = []
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        feats.append(graycoprops(glcm, prop).mean())
    return torch.tensor(feats, dtype=torch.float32).unsqueeze(0)


def get_image_path():
    root = tk.Tk()
    root.attributes('-topmost', True)
    root.withdraw()
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Pilih Gambar Biji Kopi",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    root.destroy()
    return file_path


def predict_image(model, image_path):
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    visual_input = transform(img_resized).unsqueeze(0).to(DEVICE)
    texture_input = extract_glcm(cv2.resize(img_bgr, (224, 224))).to(DEVICE)

    with torch.no_grad():
        outputs = model(visual_input, texture_input)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, idx = torch.max(prob, 0)

    class_name = CLASSES[idx.item()]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(class_name, fontsize=18, fontweight='bold')
    plt.show()


if __name__ == '__main__':
    model = FusionModel(num_classes=len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    while True:
        inp = input("ENTER untuk pilih gambar, 'q' untuk keluar: ")
        if inp.lower() == 'q':
            break

        img_path = get_image_path()
        if img_path:
            predict_image(model, img_path)
