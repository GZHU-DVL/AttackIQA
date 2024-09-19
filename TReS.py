import torch
import torchvision.models as models
import torchvision
import torch.nn.functional as F
from torch import nn, Tensor

import numpy as np
from scipy import stats
from tqdm import tqdm
import os

from transformers import Transformer
from posencode import PositionEmbeddingSine

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

class L2pooling(nn.Module):
	def __init__(self, filter_size=5, stride=1, channels=None, pad_off=0):
		super(L2pooling, self).__init__()
		self.padding = (filter_size - 2 )//2
		self.stride = stride
		self.channels = channels
		a = np.hanning(filter_size)[1:-1]
		g = torch.Tensor(a[:,None]*a[None,:])
		g = g/torch.sum(g)
		self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))
	def forward(self, input):
		input = input**2
		out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
		return (out+1e-12).sqrt()

class TReS(nn.Module):
    def __init__(self, device):
        super(TReS, self).__init__()
        self.norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.device = device


        self.L2pooling_l1 = L2pooling(channels=256)
        self.L2pooling_l2 = L2pooling(channels=512)
        self.L2pooling_l3 = L2pooling(channels=1024)
        self.L2pooling_l4 = L2pooling(channels=2048)

   
        from resnet_modify import resnet50 as resnet_modifyresnet
        dim_modelt = 3840
        modelpretrain = models.resnet50(pretrained=True)


        torch.save(modelpretrain.state_dict(), 'modelpretrain')

        self.model = resnet_modifyresnet()
        self.model.load_state_dict(torch.load('modelpretrain'), strict=True)

        self.dim_modelt = dim_modelt

        os.remove("modelpretrain")

        nheadt = 16
        num_encoder_layerst = 2
        dim_feedforwardt = 64
        ddropout = 0.5
        normalize = True

        self.transformer = Transformer(d_model=dim_modelt, nhead=nheadt,
                                       num_encoder_layers=num_encoder_layerst,
                                       dim_feedforward=dim_feedforwardt,
                                       normalize_before=normalize,
                                       dropout=ddropout)

        self.position_embedding = PositionEmbeddingSine(dim_modelt // 2, normalize=True)

        self.fc2 = nn.Linear(dim_modelt, self.model.fc.in_features)
        self.fc = nn.Linear(self.model.fc.in_features * 2, 1)

        self.ReLU = nn.ReLU()
        self.avg7 = nn.AvgPool2d((7, 7))
        self.avg8 = nn.AvgPool2d((8, 8))
        self.avg4 = nn.AvgPool2d((4, 4))
        self.avg2 = nn.AvgPool2d((2, 2))

        self.drop2d = nn.Dropout(p=0.1)
        self.consistency = nn.L1Loss()

    def forward(self, x):
        x = self.norm(x)
        self.pos_enc_1 = self.position_embedding(torch.ones(1, self.dim_modelt, 7, 7).to(self.device))
        self.pos_enc = self.pos_enc_1.repeat(x.shape[0], 1, 1, 1).contiguous()

        out, layer1, layer2, layer3, layer4 = self.model(x)

        layer1_t = self.avg8(self.drop2d(self.L2pooling_l1(F.normalize(layer1, dim=1, p=2))))
        layer2_t = self.avg4(self.drop2d(self.L2pooling_l2(F.normalize(layer2, dim=1, p=2))))
        layer3_t = self.avg2(self.drop2d(self.L2pooling_l3(F.normalize(layer3, dim=1, p=2))))
        layer4_t = self.drop2d(self.L2pooling_l4(F.normalize(layer4, dim=1, p=2)))
        layers = torch.cat((layer1_t, layer2_t, layer3_t, layer4_t), dim=1)

        out_t_c = self.transformer(layers, self.pos_enc)
        out_t_o = torch.flatten(self.avg7(out_t_c), start_dim=1)
        out_t_o = self.fc2(out_t_o)
        layer4_o = self.avg7(layer4)
        layer4_o = torch.flatten(layer4_o, start_dim=1)
        predictionQA = self.fc(torch.flatten(torch.cat((out_t_o, layer4_o), dim=1), start_dim=1))

        # =============================================================================
        # =============================================================================

        fout, flayer1, flayer2, flayer3, flayer4 = self.model(torch.flip(x, [3]))
        flayer1_t = self.avg8(self.L2pooling_l1(F.normalize(flayer1, dim=1, p=2)))
        flayer2_t = self.avg4(self.L2pooling_l2(F.normalize(flayer2, dim=1, p=2)))
        flayer3_t = self.avg2(self.L2pooling_l3(F.normalize(flayer3, dim=1, p=2)))
        flayer4_t = self.L2pooling_l4(F.normalize(flayer4, dim=1, p=2))
        flayers = torch.cat((flayer1_t, flayer2_t, flayer3_t, flayer4_t), dim=1)
        fout_t_c = self.transformer(flayers, self.pos_enc)
        fout_t_o = torch.flatten(self.avg7(fout_t_c), start_dim=1)
        fout_t_o = (self.fc2(fout_t_o))
        flayer4_o = self.avg7(flayer4)
        flayer4_o = torch.flatten(flayer4_o, start_dim=1)
        fpredictionQA = (self.fc(torch.flatten(torch.cat((fout_t_o, flayer4_o), dim=1), start_dim=1)))

        consistloss1 = self.consistency(out_t_c, fout_t_c.detach())
        consistloss2 = self.consistency(layer4, flayer4.detach())
        consistloss = 1 * (consistloss1 + consistloss2)

        return predictionQA, consistloss