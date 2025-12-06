import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics import classification_report

DATASET_ROOT = r'D:\Cool-yeah\SEMESTER 5\FinalProject_CompVis\dataset\preprocessing'
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_split_dir(root, split):
    candidates = [
        split,
        split + "_preprocessing",
        os.path.join("preprocessing", split),
        os.path.join(root, split),
        os.path.join(root, split + "_preprocessing"),
        os.path.join(root, "preprocessing", split)
    ]
    for p in candidates:
        path = p if os.path.isabs(p) else os.path.join(root, p)
        if os.path.exists(path):
            return path
    return None

def extract_glcm_features(img_gray):
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

class GLCMDataset(Dataset):
    def __init__(self, root, split):
        self.data = []
        self.classes = []
        self.split_dir = find_split_dir(root, split)
        if self.split_dir is None:
            return
        entries = [d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))]
        self.classes = sorted(entries)
        for label_idx, cls in enumerate(self.classes):
            cls_folder = os.path.join(self.split_dir, cls)
            gray_folder = os.path.join(cls_folder, 'GLCM_Grayscale')
            target = gray_folder if os.path.exists(gray_folder) else cls_folder
            if not os.path.exists(target):
                continue
            for f in os.listdir(target):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append((os.path.join(target, f), label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            img_gray = np.zeros((224, 224), dtype=np.uint8)
        feats = extract_glcm_features(img_gray)
        return feats, label

class GLCM_MLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_glcm():
    train_dir = find_split_dir(DATASET_ROOT, 'train')
    test_dir = find_split_dir(DATASET_ROOT, 'tes')
    if train_dir is None:
        print(f"Error: folder train tidak ditemukan di {DATASET_ROOT}")
        return
    if test_dir is None:
        print(f"Error: folder test tidak ditemukan di {DATASET_ROOT}")
        return

    train_ds = GLCMDataset(DATASET_ROOT, 'train')
    test_ds = GLCMDataset(DATASET_ROOT, 'tes')
    if len(train_ds) == 0 or len(test_ds) == 0:
        print("Error: dataset train atau test kosong.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = GLCM_MLP(num_classes=len(train_ds.classes)).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            out = model(feats)
            loss = loss_fn(out, labels)
            loss.backward()
            opt.step()
            _, preds = torch.max(out, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
        train_acc = 100 * correct / total if total > 0 else 0

        model.eval()
        correct_t, total_t = 0, 0
        with torch.no_grad():
            for feats, labels in test_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                out = model(feats)
                _, preds = torch.max(out, 1)
                total_t += labels.size(0)
                correct_t += (preds == labels).sum().item()
        test_acc = 100 * correct_t / total_t if total_t > 0 else 0

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    all_y, all_p = [], []
    model.eval()
    with torch.no_grad():
        for feats, labels in test_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            out = model(feats)
            _, preds = torch.max(out, 1)
            all_y.extend(labels.cpu().numpy())
            all_p.extend(preds.cpu().numpy())

    print(classification_report(all_y, all_p, target_names=train_ds.classes, digits=4))

if __name__ == "__main__":
    train_glcm()
