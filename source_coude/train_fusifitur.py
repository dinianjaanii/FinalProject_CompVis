import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from skimage.feature import graycomatrix, graycoprops
import matplotlib.pyplot as plt
import time

DATASET_ROOT = r'D:\Cool-yeah\SEMESTER 5\FinalProject_CompVis\dataset\preprocessing'
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_glcm_features(image_gray):
    glcm = graycomatrix(
        image_gray, 
        distances=[1], 
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256, 
        symmetric=True, 
        normed=True
    )
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    return torch.tensor([contrast, dissimilarity, homogeneity, energy, correlation], dtype=torch.float32)

class FusionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.split_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.data = []

        if not os.path.exists(self.split_dir):
            return

        self.classes = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])

        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.split_dir, class_name)
            rgb_folder = os.path.join(class_path, 'ShuffleNet_RGB')
            gray_folder = os.path.join(class_path, 'GLCM_Grayscale')

            if not os.path.exists(rgb_folder) or not os.path.exists(gray_folder):
                continue

            for f in os.listdir(rgb_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path_rgb = os.path.join(rgb_folder, f)
                    path_gray = os.path.join(gray_folder, f)
                    if os.path.exists(path_gray):
                        self.data.append((path_rgb, path_gray, label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_rgb, path_gray, label = self.data[idx]

        img_rgb = cv2.imread(path_rgb)
        if img_rgb is None:
            img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        img_gray = cv2.imread(path_gray, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            img_gray = np.zeros((224, 224), dtype=np.uint8)

        if self.transform:
            visual_input = self.transform(img_rgb)
        else:
            visual_input = torch.tensor(img_rgb).permute(2, 0, 1).float()

        texture_input = extract_glcm_features(img_gray)

        return visual_input, texture_input, label

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        self.cnn = models.shufflenet_v2_x1_0(weights='DEFAULT')
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

    def forward(self, x_visual, x_texture):
        feat_visual = self.cnn(x_visual)
        feat_texture = self.glcm_mlp(x_texture)
        fused = torch.cat((feat_visual, feat_texture), dim=1)
        return self.classifier(fused)

def train_fusion_no_kfold():
    train_ds = FusionDataset(DATASET_ROOT, 'train', data_transforms)
    test_ds = FusionDataset(DATASET_ROOT, 'test', data_transforms)

    if len(train_ds) == 0:
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_ds.classes)

    model = FusionModel(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history_train = []
    history_test = []

    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0

        for visuals, textures, labels in train_loader:
            visuals, textures, labels = visuals.to(DEVICE), textures.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(visuals, textures)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        train_acc = 100 * correct / total

        model.eval()
        correct_test = 0
        total_test = 0

        with torch.no_grad():
            for visuals, textures, labels in test_loader:
                visuals, textures, labels = visuals.to(DEVICE), textures.to(DEVICE), labels.to(DEVICE)
                outputs = model(visuals, textures)
                _, preds = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (preds == labels).sum().item()

        test_acc = 100 * correct_test / total_test if total_test > 0 else 0

        history_train.append(train_acc)
        history_test.append(test_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

    plt.plot(history_train, label='Train Acc')
    plt.plot(history_test, label='Test Acc')
    plt.title('Akurasi Skenario Fusi (ShuffleNet + GLCM)')
    plt.legend()
    plt.savefig('hasil_fusi_final.png')

    torch.save(model.state_dict(), 'model_fusi_kopi_final.pth')

if __name__ == '__main__':
    train_fusion_no_kfold()
