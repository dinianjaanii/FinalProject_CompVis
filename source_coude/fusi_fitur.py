import torch
import torch.nn as nn
from torchvision import models

class FusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FusionModel, self).__init__()
        self.cnn = models.shufflenet_v2_x1_0(weights='DEFAULT')
        self.cnn.fc = nn.Identity()
        
        self.glcm_mlp = nn.Sequential(
            nn.BatchNorm1d(5),
            nn.Linear(5, 32),
            nn.ReLU()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(1056, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_visual, x_texture):
        feat_visual = self.cnn(x_visual)
        feat_texture = self.glcm_mlp(x_texture)
        feat_fusion = torch.cat((feat_visual, feat_texture), dim=1)
        output = self.classifier(feat_fusion)
        return output
