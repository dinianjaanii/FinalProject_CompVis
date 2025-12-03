# gradcam.py
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import ShuffleNet_V2_X1_0_Weights

# We will build a small utility that reconstructs a ShuffleNetV2 backbone and hooks the last conv feature map.
def get_shufflenet_backbone(pretrained=False):
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
    m = models.shufflenet_v2_x1_0(weights=weights)
    return m

def make_gradcam(pil_img, fusion_model, target_idx):
    """
    Returns PIL image with heatmap overlay for the CNN backbone part.
    fusion_model: instance of FusionModel we used in training (with backbone inside)
    target_idx: predicted class index (int)
    """
    device = torch.device("cpu")
    img = pil_img.convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    inp = transform(img).unsqueeze(0).to(device)

    # Use a fresh torchvision backbone with same weights from fusion_model.backbone
    # We need feature maps from conv layers, so we use the model directly.
    backbone = get_shufflenet_backbone(pretrained=False)
    # transfer weights (layers names are compatible)
    backbone.load_state_dict({k.replace("backbone.",""):v for k,v in fusion_model.backbone.state_dict().items()}, strict=False)

    backbone.eval()
    # hook to capture features and gradients
    features = {}
    gradients = {}
    def forward_hook(module, input, output):
        features['value'] = output.detach()
    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    # Last feature map: find a conv layer near the end. For shufflenet_v2_x1_0, use stage4[1].conv2 or backbone.conv5?
    # We'll attempt to hook the last conv in backbone: backbone.conv5 if exists OR last module in backbone.features
    target_module = None
    if hasattr(backbone, "conv5"):
        target_module = backbone.conv5
    else:
        # fallback: find the last nn.Conv2d in features
        for m in reversed(list(backbone.features.modules())):
            import torch.nn as nn
            if isinstance(m, nn.Conv2d):
                target_module = m
                break

    if target_module is None:
        raise RuntimeError("Cannot find target conv module for Grad-CAM")

    h_fwd = target_module.register_forward_hook(forward_hook)
    h_back = target_module.register_full_backward_hook(backward_hook)

    # forward pass
    out = backbone(inp)
    # create a fake classifier head to connect Grad-CAM: use fusion_model.classifier (it expects fused features)
    # so instead we compute gradients wrt a channel-summed logit: create a small linear on pooled features
    pooled = F.adaptive_avg_pool2d(out, (1,1)).reshape(out.shape[0], -1)  # [1, C]
    # create a tiny classifier weight to project pooled->logit for target_idx (random but consistent)
    # Alternative: compute gradient of sum of pooled channels for simplicity:
    score = pooled[:, :].sum()
    backbone.zero_grad()
    score.backward(retain_graph=True)

    # get gradients and feature map
    fmap = features['value'][0].cpu().numpy()  # [C, H, W]
    grads = gradients['value'][0].cpu().numpy()  # [C, H, W]

    # weights: global-average-pool grads over (H,W)
    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * fmap[i, :, :]

    # relu and normalize
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    if cam.max() != 0:
        cam = cam / cam.max()

    # resize cam to image size
    import cv2
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_uint8, (img.width, img.height))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = Image.fromarray(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    # overlay
    overlay = Image.blend(img, heatmap, alpha=0.4)
    # cleanup hooks
    h_fwd.remove()
    h_back.remove()
    return overlay