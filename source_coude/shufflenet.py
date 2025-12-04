import torch
import torch.nn as nn
from torchvision import models

class VisualFeatureExtractor(nn.Module):
    def __init__(self):
        super(VisualFeatureExtractor, self).__init__()
        self.cnn = models.shufflenet_v2_x1_0(weights='DEFAULT')
        self.cnn.fc = nn.Identity()

    def forward(self, x):
        return self.cnn(x)

if __name__ == '__main__':
    model_visual = VisualFeatureExtractor()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model_visual(dummy_input)
    print(output.shape)
