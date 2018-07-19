from __future__ import print_function

import torch
#import torch.nn as nn
#from torch.nn import Linear, Conv2d, ReLU, MaxPool2d, CrossEntropyLoss
#from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import DataLoader
#import torch.nn.functional as F
#from torchvision import datasets, transforms
import numpy as np
#from torch.autograd import Variable
#import matplotlib.pyplot as plt
from load import Brats15NumpyDataset
from u_net import UNet




import json
import os.path
#%%

def dice_score(input, target):
    smooth = 1.

    iflat = input.flatten()
    tflat = target.flatten()
    intersection = np.sum(iflat * tflat)

    return ((2. * intersection + smooth) /
              (np.sum(iflat) + np.sum(tflat) + smooth))

def sensitivity_score(input, target):
    eps = 1e-15
    iflat = input.flatten()
    tflat = target.flatten()
    TP = np.sum(np.logical_and(iflat == 1, tflat == 1))
    FN = np.sum(np.logical_and(iflat == 0, tflat == 1))

    return TP/(TP+FN+eps)

def test(model, dset, n_classes, start=0, dice_start=0.0, sensitivity_start=0.0):
    # switch to evaluate mode
    model.eval()

    dloader = DataLoader(dset, batch_size=1, shuffle=False)
    dice = dice_start*(start+1)
    sensitivity = sensitivity_start*(start+1)
    #jj = 0
    #N = len(dloader)
    all_scores = {'step': [],'dice': [], 'sensitivity': []} 
    for ii, (x, y) in enumerate(dloader):
        if ii>=start:
            if n_classes == 1:
                y_pred_1h = np.sign(model(x).detach().numpy().squeeze())
                y_pred_1h[y_pred_1h<0]=0
            else:
                y_pred = np.argmax(model(x).detach().numpy(), axis=1)
                y_pred[y_pred>0] = 1 # classes 1,2,3,4 = complete tumor
                y_pred_1h = np.eye(n_classes)[y_pred] #one hot vector
                y_pred_1h = y_pred_1h.squeeze()[:,:,1] #only use class with tumor to calculate dice score
            y_1h = y.detach().numpy().squeeze()
            y_1h[y_1h>0] = 1
            #y_1h = np.eye(n_classes)[y]
            #y_1h = y_1h.squeeze()[:,:,1]
            #print(y_pred_1h.shape)
            dice_sc = dice_score(y_pred_1h,y_1h)
            dice += float(dice_sc)
            TPR = sensitivity_score(y_pred_1h,y_1h)
            sensitivity += float(TPR)
            all_scores['step'].append(ii)
            all_scores['dice'].append(float(dice_sc))
            all_scores['sensitivity'].append(float(TPR))
            if ii%100==0 or ii<10:
                print(dice_sc)
                #print('dice score so far (step {}): {}'.format(jj,dice/jj))
                #np.savez_compressed('./test_sum.npz', dice=dice, steps=jj)
                print('step {}: dice={}, sensitivity={}'.format(ii,dice/(ii+1),sensitivity/(ii+1)))
                
            if ii%500==0:
                np.savez_compressed('./test_scores.npz', dice=dice/(ii+1), steps=ii, sensitivity=sensitivity/(ii+1))
                with open('all_scores.json', 'w') as fp:
                    json.dump(all_scores, fp)
            del dice_sc, x, y, y_1h, y_pred_1h, TPR
            
    np.savez_compressed('./test_scores.npz',dice=dice/(ii+1), steps=ii,sensitivity=sensitivity/(ii+1))
    with open('all_scores.json', 'w') as fp:
        json.dump(all_scores, fp)
    #return dice, y_1h, y_pred_1h,x.detach().numpy()
    #return dice

if __name__ == '__main__':

    n_classes = 1
    use_gpu = torch.cuda.is_available()
    checkpoint_file = './training-30-7949.ckpt'
    data_file = './data/brats2015_MR_Flair_LGG_r1.h5'
    
    if 'model' not in locals(): # only reload if model doesn't exist
        model = UNet(n_classes)  
        if use_gpu:
            checkpoint = torch.load(checkpoint_file) #gpu
        else:
            checkpoint= torch.load(checkpoint_file,map_location=lambda storage, location: storage)   
    #    dset_train=Brats15NumpyDataset('./data/brats2015_MR_T2.h5', train=True, train_split=0.8, random_state=-1,
    #                 transform=None, preload_data=False, tensor_conversion=False)
    #    train(model, dset_train, n_epochs=5, batch_size=2, use_gpu=use_gpu) 
        model.load_state_dict(checkpoint)
    #test    
    dset_test=Brats15NumpyDataset(data_file , train=False, train_split=0.8, random_state=-1,
                 transform=None, preload_data=False, tensor_conversion=False)
    #dice, y, y_pred,x = test(model, dset_test, 2)
    if os.path.isfile('test_scores.npz'):
        file = np.load('test_scores.npz')
        dice_start = file['dice']
        sensitivity_start = file['sensitivity']
        step = file['steps']
        test(model, dset_test, 1, start=step+1, dice_start=dice_start, sensitivity_start=sensitivity_start)
    else:
        test(model, dset_test, 1)
#%%
        
#file = np.load('./saved/test_scores_5epochs.npz')
#dice_start = file['dice']
#sensitivity_start = file['sensitivity']