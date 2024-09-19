import os
from ImageDataset import ImageDataset
import torch
import torchvision
import torch.nn as nn
from SCNN import SCNN
from PIL import Image
from scipy import stats
import random
import torch.nn.functional as F
import numpy as np
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]
class DBCNN(torch.nn.Module):

    def __init__(self, scnn_root = 'pretrained_scnn/scnn.pkl'):
        """Declare all needed layers."""
        nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features1 = torchvision.models.vgg16(pretrained=True).features
        self.features1 = nn.Sequential(*list(self.features1.children())
        [:-1])
        scnn = SCNN()
        scnn = torch.nn.DataParallel(scnn).cuda()

        scnn.load_state_dict(torch.load(scnn_root))
        self.features2 = scnn.module.features

        # Linear classifier.
        self.fc = torch.nn.Linear(512 * 128, 1)
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def forward(self, X):
        """Forward pass of the network.
        """
        X = self.norm(X)
        N = X.size()[0]
        X1 = self.features1(X)
        H = X1.size()[2]
        W = X1.size()[3]
        assert X1.size()[1] == 512
        X2 = self.features2(X)
        H2 = X2.size()[2]
        W2 = X2.size()[3]
        assert X2.size()[1] == 128

        if (H != H2) | (W != W2):
            X2 = F.upsample_bilinear(X2, (H, W))

        X1 = X1.view(N, 512, H * W)
        X2 = X2.view(N, 128, H * W)
        X = torch.bmm(X1, torch.transpose(X2, 1, 2)) / (H * W)  # Bilinear
        assert X.size() == (N, 512, 128)
        X = X.view(N, 512 * 128)
        X = torch.sqrt(X + 1e-8)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, 1)
        return X



