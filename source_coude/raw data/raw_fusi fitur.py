import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report

DATASET_ROOT = r'/content/drive/MyDrive/FinalProject_CompVis/dataset/mentah'
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_glcm(img_gray):
    glcm = graycomatrix(img_gray, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

    feats = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
    ]
    return torch.tensor(feats, dtype=torch.float32)

class FusionDataset(Dataset):
    def __init__(self, root, split, transform):
        self.transform = transform
        self.data = []
        self.split_dir = os.path.join(root, split)
        self.classes = sorted(os.listdir(self.split_dir))

        for label_idx, cls in enumerate(self.classes):
            folder = os.path.join(self.split_dir, cls)
            rgb_dir = os.path.join(folder, "ShuffleNet_RGB")
            gray_dir = os.path.join(folder, "GLCM_Grayscale")

            if not os.path.exists(rgb_dir) or not os.path.exists(gray_dir):
                continue

            for f in os.listdir(rgb_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    rgb_path = os.path.join(rgb_dir, f)
                    gray_path = os.path.join(gray_dir, f)
                    if os.path.exists(gray_path):
                        self.data.append((rgb_path, gray_path, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        f_rgb, f_gray, label = self.data[idx]

        img_rgb = cv2.cvtColor(cv2.imread(f_rgb), cv2.COLOR_BGR2RGB)
        img_gray = cv2.imread(f_gray, cv2.IMREAD_GRAYSCALE)

        visual = self.transform(img_rgb)
        texture = extract_glcm(img_gray)

        return visual, texture, label


transform_rgb = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = models.shufflenet_v2_x1_0(weights="DEFAULT")
        self.cnn.fc = nn.Identity()

        self.glcm_mlp = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Linear(5, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, glcm):
        fv = self.cnn(rgb)
        ft = self.glcm_mlp(glcm)
        fusion = torch.cat((fv, ft), dim=1)
        return self.classifier(fusion)

def train_fusion():
    train_ds = FusionDataset(DATASET_ROOT, 'train', transform_rgb)
    test_ds = FusionDataset(DATASET_ROOT, 'test', transform_rgb)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionModel(num_classes=len(train_ds.classes)).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0

        for rgb, glcm, labels in train_loader:
            rgb, glcm, labels = rgb.to(DEVICE), glcm.to(DEVICE), labels.to(DEVICE)

            opt.zero_grad()
            outputs = model(rgb, glcm)
            loss = loss_fn(outputs, labels)
            loss.backward()
            opt.step()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        correct_t, total_t = 0, 0
        with torch.no_grad():
            for rgb, glcm, labels in test_loader:
                rgb, glcm, labels = rgb.to(DEVICE), glcm.to(DEVICE), labels.to(DEVICE)
                out = model(rgb, glcm)
                _, preds = torch.max(out, 1)
                total_t += labels.size(0)
                correct_t += (preds == labels).sum().item()

        test_acc = 100 * correct_t / total_t
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}%")

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for rgb, glcm, labels in test_loader:
            rgb, glcm, labels = rgb.to(DEVICE), glcm.to(DEVICE), labels.to(DEVICE)
            out = model(rgb, glcm)
            _, preds = torch.max(out, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    print("\n=== LAPORAN FUSI ===")
    print(classification_report(y_true, y_pred, target_names=train_ds.classes, digits=4))

if __name__ == "__main__":
    train_fusion()