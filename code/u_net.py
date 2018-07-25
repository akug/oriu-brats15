from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, normal_
import torch.nn.functional as F
import numpy as np


#%%


def initialize_weights(*models):
    ''' Initialize weights of a model.
    '''
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                #nn.init.kaiming_normal_(module.weight)
                # or:
                xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
    
    
class DownConv(nn.Module):
    ''' Perform two 3x3 convolutions (with ReLU activation)
    and a 2x2 max pooling operation
    Use batch normalisation to speed up training.
    '''
    def __init__(self, in_channels, out_channels, dropout=False):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        layers = [
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.downconv = nn.Sequential(*layers)
          
    def forward(self, x):
        return self.downconv(x)
    
    
class UpConv(nn.Module):
    ''' Perform two 3x3 convolutions (with ReLU activation)
    and a 2x2 max pooling operation
    Use Batch Normalisation to speed up training.
    '''  
    def __init__(self, in_channels, middle_channels, out_channels):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        
        self.upconv = nn.Sequential(               
                nn.Conv2d(self.in_channels, self.middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.middle_channels, self.middle_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(middle_channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2)
        )
          
    def forward(self, x):
        return self.upconv(x)
    
    
class UNet(nn.Module):
    ''' U-Net with depth 4
    based on: https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/u_net.py
    and https://github.com/jaxony/unet-pytorch/blob/master/model.py
    '''
    def __init__(self, n_classes):
        '''
        Arguments:
            n_classes: number of classes
        '''
        super(UNet, self).__init__()

        self.down1 = DownConv(1, 64)
        self.down2 = DownConv(64, 128)
        self.down3 = DownConv(128, 256, dropout=True)
        self.center = UpConv(256, 512, 256)
        self.up3 = UpConv(512, 256, 128)
        self.up2 = UpConv(256, 128, 64)
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(64, n_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        center = self.center(down3)
        up3 = self.up3(torch.cat([center, F.upsample(down3, center.size()[2:], mode='bilinear', align_corners=True)], 1))
        up2 = self.up2(torch.cat([up3, F.upsample(down2, up3.size()[2:], mode='bilinear', align_corners=True)], 1))
        up1 = self.up1(torch.cat([up2, F.upsample(down1, up2.size()[2:], mode='bilinear', align_corners=True)], 1))
        final = self.final(up1)
        return F.upsample(final, x.size()[2:], mode='bilinear',align_corners=True)
    

#if __name__ == '__main__':
#
#    n_classes = 2
#    model = UNet(n_classes)  

