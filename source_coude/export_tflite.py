# export_tflite.py
"""
Export pipeline:
1) Export the FusionModel to ONNX using a sample input (image tensor + glcm tensor).
2) Then convert ONNX -> TFLite (requires external tools).
Notes: direct PyTorch -> TFLite is not officially supported; these steps provide a standard path.
"""

import torch
from PIL import Image
import numpy as np
from glcm_shufflenet import FusionModel, compute_glcm_features

MODEL_PATH = "best_model.pth"
IMG_SIZE = 224

def load_model():
    ck = torch.load(MODEL_PATH, map_location="cpu")
    class_to_idx = ck["class_to_idx"]
    dummy = Image.new("RGB",(128,128),(128,128,128))
    glcm_dim = compute_glcm_features(dummy, distances=[1,2,3]).shape[0]
    model = FusionModel(glcm_dim=glcm_dim, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model

def export_onnx(out_path="fusion_model.onnx"):
    model = load_model()
    # dummy inputs
    img_dummy = torch.randn(1,3,IMG_SIZE,IMG_SIZE)
    glcm_dummy = torch.randn(1, model.classifier[0].in_features - model.backbone.fc.in_features) if False else torch.randn(1, model.classifier[0].in_features - model.backbone.fc.in_features)
    # Can't easily compute dims this way due to identity; better to compute glcm_dim directly:
    glcm_dummy = torch.randn(1, compute_glcm_features(Image.new("RGB",(128,128),(128,128,128)), distances=[1,2,3]).shape[0])

    # Export
    torch.onnx.export(model,
                      (img_dummy, glcm_dummy),
                      out_path,
                      opset_version=12,
                      input_names=["image","glcm"],
                      output_names=["logits"],
                      dynamic_axes={"image":{0:"batch"}, "glcm":{0:"batch"}, "logits":{0:"batch"}})
    print("Saved ONNX:", out_path)

if __name__ == "__main__":
    export_onnx()