# batch_inference.py
import os, shutil, csv
from PIL import Image
import torch
from torchvision import transforms
from glcm_shufflenet import FusionModel, compute_glcm_features

MODEL_PATH = "best_model.pth"
GLCM_DISTANCES = [1,2,3]
IMG_SIZE = 224

def load_model():
    ck = torch.load(MODEL_PATH, map_location="cpu")
    class_to_idx = ck["class_to_idx"]
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    dummy = Image.new("RGB",(128,128),(128,128,128))
    glcm_dim = compute_glcm_features(dummy, distances=GLCM_DISTANCES).shape[0]
    model = FusionModel(glcm_dim=glcm_dim, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, idx_to_class

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def run_folder(folder_path, out_csv="batch_pred.csv", move_to_class_folders=False, dest_root=None):
    model, idx_to_class = load_model()
    rows = []
    files = [os.path.join(folder_path,f) for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".jpeg",".png",".bmp",".tiff"))]
    for fp in files:
        img = Image.open(fp).convert("RGB")
        glcm = compute_glcm_features(img.resize((128,128)), distances=GLCM_DISTANCES)
        glcm_t = torch.from_numpy(glcm).unsqueeze(0)
        img_t = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(img_t, glcm_t)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]
            pred_idx = int(out.argmax(dim=1).item())
            pred_label = idx_to_class[pred_idx]
            score = float(probs[pred_idx])
        rows.append([fp, pred_label, score])
        if move_to_class_folders and dest_root is not None:
            target_dir = os.path.join(dest_root, pred_label)
            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(fp, os.path.join(target_dir, os.path.basename(fp)))
    with open(out_csv, "w", newline="", encoding="utf8") as f:
        writer = csv.writer(f)
        writer.writerow(["file","pred","score"])
        writer.writerows(rows)
    print(f"Saved {len(rows)} results to {out_csv}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--folder", required=True)
    p.add_argument("--out", default="batch_pred.csv")
    p.add_argument("--move", action="store_true")
    p.add_argument("--dest", default=None)
    args = p.parse_args()
    run_folder(args.folder, out_csv=args.out, move_to_class_folders=args.move, dest_root=args.dest)