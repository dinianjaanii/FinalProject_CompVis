import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import classification_report
import numpy as np
import time
import matplotlib.pyplot as plt

DATASET_ROOT = r'/content/drive/MyDrive/FinalProject_CompVis/dataset/mentah'
BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ShuffleNetDataset(Dataset):
    def __init__(self, root_dir, split_type, transform=None):
        self.transform = transform
        self.data = []
        candidates = [
            split_type,
            split_type + "_preprocessing",
            "preprocessing/" + split_type
        ]
        self.split_dir = None
        for path in candidates:
            full_path = os.path.join(root_dir, path)
            if os.path.exists(full_path):
                self.split_dir = full_path
                break
        if self.split_dir is None:
            return
        self.classes = sorted([
            d for d in os.listdir(self.split_dir)
            if os.path.isdir(os.path.join(self.split_dir, d))
        ])
        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.split_dir, class_name)
            rgb_folder = os.path.join(class_path, 'ShuffleNet_RGB')
            target_folder = rgb_folder if os.path.exists(rgb_folder) else class_path
            for f in os.listdir(target_folder):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.data.append((os.path.join(target_folder, f), label_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_img, label = self.data[idx]
        img = cv2.imread(path_img)
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

class ShuffleNetOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super(ShuffleNetOnlyModel, self).__init__()
        self.cnn = models.shufflenet_v2_x1_0(weights='DEFAULT')
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.cnn(x)

def train_shufflenet():
        train_ds = ShuffleNetDataset(DATASET_ROOT, 'train', data_transforms)
        test_ds = ShuffleNetDataset(DATASET_ROOT, 'test', data_transforms)

        if len(train_ds) == 0 or len(test_ds) == 0:
            return

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = ShuffleNetOnlyModel(num_classes=len(train_ds.classes)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        history = {'train_acc': [], 'test_acc': []}
        start_time = time.time()

        for epoch in range(EPOCHS):
            model.train()
            correct = 0
            total = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
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
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    total_test += labels.size(0)
                    correct_test += (preds == labels).sum().item()

            test_acc = 100 * correct_test / total_test if total_test > 0 else 0
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)

            print(f"Epoch {epoch+1}/{EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

        torch.save(model.state_dict(), 'model_shufflenet_only.pth')

        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(
            all_labels,
            all_preds,
            target_names=train_ds.classes,
            digits=4
        )
        print(report)

if __name__ == '__main__':
    train_shufflenet()