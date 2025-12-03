import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse
import time
import copy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from skimage import img_as_ubyte

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------
# CONFIG
# ---------------------------
parser = argparse.ArgumentParser(description="Train GLCM + ShuffleNetV2 fusion classifier")
parser.add_argument('--data_root', type=str, default='dataset', help='Root dataset folder (contains class subfolders)')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--glcm_distances', nargs='+', type=int, default=[1,2,3])  # distances for GLCM
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--save_path', type=str, default='best_model.pth')
parser.add_argument('--pretrained', action='store_true', help='Use ImageNet pretrained ShuffleNetV2')
args = parser.parse_args([])  # empty list for notebook style; remove to use CLI

# ---------------------------
# Utility: compute GLCM features
# ---------------------------
def compute_glcm_features(pil_image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """
    Input: PIL Image (RGB or grayscale)
    Output: 1D numpy array of GLCM features
    Approach: convert to grayscale uint8, compute GLCM for given distances & angles,
              compute props ['contrast','dissimilarity','homogeneity','energy','correlation','ASM'],
              aggregate by mean across angles for each distance, and then flatten across distances.
    """
    # convert to grayscale float and to uint8 (0..255)
    img = np.array(pil_image)
    if img.ndim == 3:
        gray = rgb2gray(img)  # floats 0..1
    else:
        gray = img.astype(np.float32) / 255.0
    gray_u8 = img_as_ubyte(gray)  # uint8 0..255

    props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    feature_list = []

    # quantize levels if needed (we keep 256 levels); skimage will accept uint8
    levels = 256

    for d in distances:
        glcm = graycomatrix(gray_u8, distances=[d], angles=angles, levels=levels, symmetric=True, normed=True)
        # glcm shape: levels x levels x len(distances=1) x len(angles)
        feats_for_d = []
        for p in props:
            arr = graycoprops(glcm, p).flatten()  # returns shape (len(distances)*len(angles),)
            # aggregate across angles by mean (we already have distances=1 in glcm call)
            mean_val = np.mean(arr)
            feats_for_d.append(mean_val)
        # feats_for_d length = len(props) for this distance
        feature_list.extend(feats_for_d)

    return np.array(feature_list, dtype=np.float32)  # length = len(distances) * len(props)

# ---------------------------
# Dataset
# ---------------------------
class CoffeeDefectDataset(Dataset):
    """
    Expects folder structure:
    data_root/
      classA/
        train/   (images)
        test/    (images)
      classB/
    We will build datasets from 'mode' subfolders ('train' or 'test').
    """
    def __init__(self, data_root, mode='train', img_size=224, glcm_distances=[1,2,3], transform=None):
        super().__init__()
        self.samples = []
        self.transform = transform
        self.img_size = img_size
        self.glcm_distances = glcm_distances

        classes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        classes.sort()
        self.class_to_idx = {c: i for i, c in enumerate(classes)}

        for c in classes:
            folder = os.path.join(data_root, c, mode)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    self.samples.append((os.path.join(folder, fname), self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = Image.open(path).convert('RGB')

        # compute GLCM features on a resized grayscale (for stability & speed)
        glcm_input = pil.resize((128,128))
        glcm_feats = compute_glcm_features(glcm_input, distances=args.glcm_distances)

        # image transform
        if self.transform:
            img_tensor = self.transform(pil)
        else:
            img_tensor = transforms.ToTensor()(pil)

        return img_tensor, torch.from_numpy(glcm_feats), label

# ---------------------------
# Model: ShuffleNetV2 backbone + fusion MLP
# ---------------------------
class FusionModel(nn.Module):
    def __init__(self, glcm_dim, num_classes, pretrained=False):
        super().__init__()
        # Load ShuffleNetV2 (x1.0)
        self.backbone = models.shufflenet_v2_x1_0(pretrained=pretrained)
        # find feature dim (fc.in_features)
        if hasattr(self.backbone, 'fc'):
            feat_dim = self.backbone.fc.in_features
            # replace classifier with identity (we'll use pooled features)
            self.backbone.fc = nn.Identity()
        else:
            # fallback
            feat_dim = 1024

        # Fusion classifier
        fused_dim = feat_dim + glcm_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_img, x_glcm):
        # x_img: [B,3,H,W]
        feat_img = self.backbone(x_img)  # expect [B, feat_dim]
        if feat_img.dim() > 2:
            feat_img = torch.flatten(feat_img, 1)
        # ensure glcm is float tensor
        feat_glcm = x_glcm.float()
        if feat_glcm.dim() == 1:
            feat_glcm = feat_glcm.unsqueeze(0)
        # fuse
        fused = torch.cat([feat_img, feat_glcm], dim=1)
        out = self.classifier(fused)
        return out

# ---------------------------
# Training and Evaluation Helpers
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for imgs, glcms, labels in tqdm(loader, desc="Train batches", leave=False):
        imgs = imgs.to(device)
        glcms = glcms.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs, glcms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, glcms, labels in tqdm(loader, desc="Val/Test batches", leave=False):
            imgs = imgs.to(device)
            glcms = glcms.to(device)
            labels = labels.to(device)

            outputs = model(imgs, glcms)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, all_labels, all_preds

# ---------------------------
# Main training flow
# ---------------------------
def main():
    device = torch.device(args.device)
    print("Device:", device)

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # Build datasets
    train_dataset = CoffeeDefectDataset(args.data_root, mode='train', img_size=args.img_size,
                                       glcm_distances=args.glcm_distances, transform=train_transforms)
    test_dataset = CoffeeDefectDataset(args.data_root, mode='test', img_size=args.img_size,
                                      glcm_distances=args.glcm_distances, transform=test_transforms)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        raise RuntimeError("Train or test dataset is empty. Pastikan folder 'train' dan 'test' ada di setiap kelas.")

    num_classes = len(train_dataset.class_to_idx)
    print("Classes:", train_dataset.class_to_idx)

    # compute GLCM feature dim
    sample_glcm = compute_glcm_features(Image.new('RGB', (128,128), color=(128,128,128)), distances=args.glcm_distances)
    glcm_dim = sample_glcm.shape[0]
    print("GLCM feature dimension:", glcm_dim)

    # dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # model
    model = FusionModel(glcm_dim=glcm_dim, num_classes=num_classes, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, y_true, y_pred = eval_model(model, test_loader, criterion, device)
        elapsed = time.time() - t0
        print(f" Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f" Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f} | time: {elapsed:.1f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({'model_state': best_model_wts,
                        'class_to_idx': train_dataset.class_to_idx},
                       args.save_path)
            print(f" Saved best model to {args.save_path} (acc {best_acc:.4f})")

    # final report
    print("\n=== Training finished ===")
    model.load_state_dict(best_model_wts)
    _, _, y_true, y_pred = eval_model(model, test_loader, criterion, device)
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=list(train_dataset.class_to_idx.keys())))

if __name__ == "__main__":
    main()