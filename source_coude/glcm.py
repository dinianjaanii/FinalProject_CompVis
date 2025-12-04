import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import torch
import torch.nn as nn

class TextureFeatureExtractor(nn.Module):
    def __init__(self):
        super(TextureFeatureExtractor, self).__init__()
        self.distances = [1]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        self.levels = 256
        
    def compute_glcm_features(self, image_gray):
        glcm = graycomatrix(image_gray,
                            distances=self.distances,
                            angles=self.angles,
                            levels=self.levels,
                            symmetric=True,
                            normed=True)
        
        contrast = graycoprops(glcm, 'contrast').mean()
        dissimilarity = graycoprops(glcm, 'dissimilarity').mean()
        homogeneity = graycoprops(glcm, 'homogeneity').mean()
        energy = graycoprops(glcm, 'energy').mean()
        correlation = graycoprops(glcm, 'correlation').mean()
        
        return torch.tensor([contrast, dissimilarity, homogeneity, energy, correlation], dtype=torch.float32)

    def forward(self, x_gray_batch):
        batch_features = []
        for img in x_gray_batch:
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy().astype(np.uint8)
            features = self.compute_glcm_features(img)
            batch_features.append(features)
        return torch.stack(batch_features)
