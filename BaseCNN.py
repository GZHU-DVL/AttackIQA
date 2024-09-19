import torch.nn as nn
from torchvision import models
from BCNN import BCNN
import torch

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

class BaseCNN(nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        self.backbone = models.resnet34(pretrained=True)
        outdim = 2
        self.representation = BCNN()
        self.fc = nn.Linear(512 * 512, outdim)
    
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



    def forward(self, x):
        """Forward pass of the network.
        """
        x = self.norm(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

    
        x = self.representation(x)
  

        x = self.fc(x)

        mean = x[:, 0]
        t = x[:, 1]
        var = nn.functional.softplus(t)
        return mean, var
