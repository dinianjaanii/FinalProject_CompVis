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
import matplotlib.pyplot as plt
import time

# ================= KONFIGURASI (SESUAI FOLDER ANDA) =================
# Arahkan ke folder induk dataset Anda
# Contoh: 'dataset' (jika di dalamnya ada train_preprocessing & test_preprocessing)
DATASET_ROOT = r'D:\Cool-yeah\SEMESTER 5\FinalProject_CompVis' 

BATCH_SIZE = 16
LEARNING_RATE = 0.0001
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. FUNGSI EKSTRAKSI GLCM =================
def extract_glcm_features(image_gray):
    # Hitung GLCM
    glcm = graycomatrix(image_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast').mean()
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    energy = graycoprops(glcm, 'energy').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    
    return torch.tensor([contrast, dissimilarity, homogeneity, energy, correlation], dtype=torch.float32)

# ================= 2. DATASET LOADER (DISESUAIKAN DENGAN STRUKTUR ANDA) =================
class FusionDataset(Dataset):
    def __init__(self, root_dir, split_folder_name, transform=None):
        """
        root_dir: Folder utama (misal: 'dataset')
        split_folder_name: Nama folder split (misal: 'train_preprocessing' atau 'test_preprocessing')
        """
        # Path lengkap ke folder split (misal: dataset/train_preprocessing)
        self.split_dir = os.path.join(root_dir, split_folder_name)
        self.transform = transform
        self.data = []
        
        print(f"\n[DEBUG] Memuat dataset dari: {self.split_dir}")

        if not os.path.exists(self.split_dir):
            print(f"❌ Error: Folder '{self.split_dir}' tidak ditemukan!")
            return

        # Ambil daftar kelas (berjamur, berlubang, dll)
        self.classes = sorted([d for d in os.listdir(self.split_dir) if os.path.isdir(os.path.join(self.split_dir, d))])
        
        if len(self.classes) == 0:
            print(f"❌ Error: Tidak ada folder kelas di dalam {self.split_dir}")
            return
            
        print(f"   -> Kelas ditemukan: {self.classes}")

        for label_idx, class_name in enumerate(self.classes):
            # Masuk ke folder kelas: dataset/train_preprocessing/berjamur
            class_path = os.path.join(self.split_dir, class_name)
            
            # Cari sub-folder RGB dan Gray di dalamnya
            rgb_folder = os.path.join(class_path, 'ShuffleNet_RGB')
            gray_folder = os.path.join(class_path, 'GLCM_Grayscale')
            
            # Validasi keberadaan folder
            if not os.path.exists(rgb_folder):
                # Coba cari nama folder lain (mungkin huruf kecil/besar beda)
                # Atau jika Anda menamainya 'RGB' saja, ganti baris ini
                print(f"   ⚠️ Warning: Folder 'ShuffleNet_RGB' tidak ada di kelas {class_name}")
                continue
                
            if not os.path.exists(gray_folder):
                print(f"   ⚠️ Warning: Folder 'GLCM_Grayscale' tidak ada di kelas {class_name}")
                continue
            
            # Ambil file gambar dari folder RGB
            files = os.listdir(rgb_folder)
            count = 0
            for f in files:
                if not f.lower().endswith(('.png', '.jpg', '.jpeg')): continue

                path_rgb = os.path.join(rgb_folder, f)
                path_gray = os.path.join(gray_folder, f)
                
                # Pastikan file pasangannya ada di folder Gray
                if os.path.exists(path_gray):
                    self.data.append((path_rgb, path_gray, label_idx))
                    count += 1
            
            print(f"   - Kelas {class_name}: {count} gambar valid.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path_rgb, path_gray, label = self.data[idx]
        
        try:
            # Load Gambar
            img_rgb = cv2.imread(path_rgb)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            
            img_gray = cv2.imread(path_gray, cv2.IMREAD_GRAYSCALE)
            
            # Transformasi
            if self.transform:
                visual_input = self.transform(img_rgb)
            else:
                visual_input = torch.tensor(img_rgb).permute(2,0,1).float()
                
            texture_input = extract_glcm_features(img_gray)
            
            return visual_input, texture_input, label
            
        except Exception as e:
            print(f"Error loading: {path_rgb} | {e}")
            return torch.zeros(3, 224, 224), torch.zeros(5), label

# ================= 3. ARSITEKTUR MODEL FUSI =================
class FusionModel(nn.Module):
    def __init__(self, num_classes):
        super(FusionModel, self).__init__()
        # Visual Branch (ShuffleNet)
        self.cnn = models.shufflenet_v2_x1_0(weights='DEFAULT')
        self.cnn.fc = nn.Identity() 
        
        # Texture Branch (GLCM MLP)
        self.glcm_mlp = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Linear(5, 32),
            nn.ReLU()
        )
        
        # Fusion & Classification
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_visual, x_texture):
        feat_visual = self.cnn(x_visual)
        feat_texture = self.glcm_mlp(x_texture)
        feat_fusion = torch.cat((feat_visual, feat_texture), dim=1)
        return self.classifier(feat_fusion)

# ================= 4. TRAINING LOOP =================
def train_model():
    # --- UPDATE NAMA FOLDER DISINI ---
    # Sesuaikan dengan nama folder asli Anda di Windows Explorer
    TRAIN_FOLDER_NAME = 'dataset/train_preprocessing' 
    TEST_FOLDER_NAME = 'dataset/test_preprocessing'  # Pastikan Anda juga punya folder ini untuk testing
    
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Data
    print("🚀 Memuat Dataset...")
    train_ds = FusionDataset(DATASET_ROOT, TRAIN_FOLDER_NAME, data_transforms)
    
    # Cek apakah folder test ada, jika tidak ada, pakai train dulu (tapi sebaiknya ada test)
    if os.path.exists(os.path.join(DATASET_ROOT, TEST_FOLDER_NAME)):
        test_ds = FusionDataset(DATASET_ROOT, TEST_FOLDER_NAME, data_transforms)
    else:
        print("⚠️ Folder Test tidak ditemukan! Menggunakan sebagian data train sebagai validasi (darurat).")
        # Split manual darurat (opsional)
        train_size = int(0.8 * len(train_ds))
        test_size = len(train_ds) - train_size
        train_ds, test_ds = torch.utils.data.random_split(train_ds, [train_size, test_size])
    
    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Dapatkan jumlah kelas otomatis
    # Jika train_ds adalah Subset (hasil random_split), akses dataset aslinya
    if isinstance(train_ds, torch.utils.data.Subset):
        num_classes = len(train_ds.dataset.classes)
        class_names = train_ds.dataset.classes
    else:
        num_classes = len(train_ds.classes)
        class_names = train_ds.classes
        
    print(f"📊 Total Kelas: {num_classes}")

    # Init Model
    model = FusionModel(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    print(f"\n🚀 Mulai Training ({EPOCHS} Epochs)...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for visuals, textures, labels in train_loader:
            visuals, textures, labels = visuals.to(DEVICE), textures.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(visuals, textures)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        # Evaluasi
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
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.1f}% | Test Acc: {test_acc:.1f}%")

    print(f"\n✅ Selesai dalam {(time.time()-start_time)/60:.1f} menit.")
    torch.save(model.state_dict(), 'model_fusi_kopi_custom.pth')
    
    # Grafik
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['test_acc'], label='Test')
    plt.title('Akurasi')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], color='orange')
    plt.title('Loss')
    plt.savefig('grafik_training_custom.png')

    # Classification Report
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for visuals, textures, labels in test_loader:
            visuals, textures = visuals.to(DEVICE), textures.to(DEVICE)
            outputs = model(visuals, textures)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\n--- LAPORAN AKHIR ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

if __name__ == '__main__':
    train_model()