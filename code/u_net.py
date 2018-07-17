from __future__ import print_function

import torch
import torch.nn as nn
#from torch.nn import Linear, Conv2d, ReLU, MaxPool2d, CrossEntropyLoss
from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from torchvision import datasets, transforms
import numpy as np
from torch.autograd import Variable
#import matplotlib.pyplot as plt

try:
    from tqdm import tqdm, trange
    print_fn = tqdm.write
    has_tqdm = True
except ImportError:
    print_fn = print
    has_tqdm = False


from load import Brats15NumpyDataset

#%%
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    
    tflat = target.view(-1)
    
    intersection = (iflat * tflat).sum()

    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))
    
def dice_score(input, target):
    smooth = 1.

    iflat = input.flatten()
    tflat = target.flatten()
    intersection = np.sum(iflat * tflat)

    return ((2. * intersection + smooth) /
              (np.sum(iflat) + np.sum(tflat) + smooth))


def dice_coef(y_true, y_pred):
    y_true_f = np.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)



def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                #nn.init.kaiming_normal_(module.weight)
                # or:
                xavier_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
                
                # normal_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
                
  
    
    
class DownConv(nn.Module):
    # perfoms 2 3x3 convolutions (with ReLU) and a 2x2 max pooling operation
    # use Batch Normalisation to speed up training
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
    # perfoms 2 3x3 convolutions (with ReLU) and a 2x2 max pooling operation
    # use Batch Normalisation to speed up training
    # based on: https://github.com/zijundeng/pytorch-semantic-segmentation/blob/master/models/u_net.py
    # and https://github.com/jaxony/unet-pytorch/blob/master/model.py
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
    # UNet with depth 4
    def __init__(self, n_classes):
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
    
def train(model, dset, n_epochs=10, batch_size=2, use_gpu=True):

    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    if use_gpu:
        model.cuda()
        
    Loss = nn.CrossEntropyLoss()
    # TODO: use different loss (e.g. BCE -> one hot encoded y or Dice)
    Optimizer = torch.optim.Adam(model.parameters()) 
    for e in range(n_epochs):
        for e_step, (x, y) in enumerate(dloader):
            print(e_step)
            print(y)
            train_step = e_step + len(dloader)*e
            #print(x.size())
            #x = x.numpy()
            #y = y.numpy().astype(np.int32)

            #if x.ndim == 2:
            #    x = x[None, :]
            #x = torch.from_numpy(x)
            #y = torch.from_numpy(y)
            y = y.long()
                
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
                
            #y = y.squeeze() 
            # Forward
            #print('forward')
            prediction = model(x)
            #pred_probs = F.sigmoid(prediction)
            #pred_probs_flat = pred_probs.view(-1)
            
            #y_flat = y.view(-1)
            # Loss
            loss = Loss(prediction, y.squeeze(1))
            #print('acc')
		            #acc = torch.mean(torch.eq(torch.argmax(prediction, dim=-1),y).float())
            Optimizer.zero_grad()
            # Backward
            #print('backward')
            loss.backward()              
            # Update
            Optimizer.step()   
            if train_step % 25 == 0:
#                print('{}: Batch-Accuracy = {}, Loss = {}'\
#                          .format(train_step, float(acc), float(loss)))
            	print('{}: Loss = {}'.format(train_step, float(loss)))
        if train_step % 10000 == 0:
             torch.save(model.state_dict(), 'training-{}-{}.ckpt'.format(e,train_step))

        checkpoint = {
            'epoch': e + 1,
            'state_dict': model.state_dict(),
            'optimizer' : Optimizer.state_dict(),
        }
        torch.save(checkpoint, 'unet1024-{}'.format(e+1))

def test(model, dset, n_classes):

    dloader = DataLoader(dset, batch_size=1, shuffle=False)
    #sums = [0.0, 0.0]
    dice = 0
    #jj = 0
    N = len(dloader)
    for ii, (x, y) in enumerate(dloader):
        y_pred = np.argmax(model(x).detach().numpy(), axis=1)
        y_pred[y_pred>0] = 1
        y_pred_1h = np.eye(n_classes)[y_pred] #one hot vector
        y_pred_1h = y_pred_1h.squeeze()[:,:,1]
        y = y.detach().numpy()
        y[y>0] = 1
        y_1h = np.eye(n_classes)[y]
        y_1h = y_1h.squeeze()[:,:,1]
        dice_sc = dice_score(y_pred_1h,y_1h)
        dice += dice_sc
        #jj += 1
        if ii%400==0:
            print(dice_sc)
            #print('dice score so far (step {}): {}'.format(jj,dice/jj))
            #np.savez_compressed('./test_sum.npz', dice=dice, steps=jj)
            print('dice score so far (step {}): {}'.format(ii,dice/(ii+1)))
        if ii%1000==0:
            np.savez_compressed('./test_dice.npz', dice=dice/(ii+1), steps=ii)


    np.savez_compressed('./test_dice.npz',dice=dice/N, steps=ii)
    #return dice, y_1h, y_pred_1h,x.detach().numpy()
    return dice

if __name__ == '__main__':

    n_classes = 5
    use_gpu = torch.cuda.is_available()
    if 'model' not in locals(): # only reload if model doesn't exist
        model = UNet(n_classes)  
        if use_gpu:
            checkpoint = torch.load('training-10.ckpt') #gpu
        else:
            checkpoint= torch.load('training-10.ckpt',map_location=lambda storage, location: storage)   
    #    dset_train=Brats15NumpyDataset('./data/brats2015_MR_T2.h5', train=True, train_split=0.8, random_state=-1,
    #                 transform=None, preload_data=False, tensor_conversion=False)
    #    train(model, dset_train, n_epochs=5, batch_size=2, use_gpu=use_gpu) 
        model.load_state_dict(checkpoint)
    #test    
    dset_test=Brats15NumpyDataset('./data/brats2015_MR_T2.h5', train=False, train_split=0.8, random_state=-1,
                 transform=None, preload_data=False, tensor_conversion=False)
    #dice, y, y_pred,x = test(model, dset_test, 2)
    dice = test(model, dset_test, 2)
    

#%%    
##############################    
##   test using random tensor:
#    start = time.time()
#    x = Variable(torch.FloatTensor(np.random.random((1, 3, 80, 80))))
#    out = model(x)
#    loss = torch.sum(out)
#    loss.backward()
#    end = time.time()
#    print(end - start)
    
#%%
#import matplotlib.pyplot as plt
#y = y.squeeze()
#y_pred = y_pred.squeeze()
#x = x.squeeze()
#plt.figure()
#plt.subplot(1,3,1)
#plt.imshow(np.argmax(y,axis=2))
#plt.subplot(1,3,2)
#plt.imshow(np.argmax(y_pred,axis=2))
#plt.subplot(1,3,3)
#plt.imshow(x)
#
