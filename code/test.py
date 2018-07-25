from __future__ import print_function

import torch
from torch.utils.data import DataLoader
import numpy as np
from load import Brats15NumpyDataset
from u_net import UNet

import json
import os.path
#%%

def dice_score(input, target):
    ''' Calculate Dice score
    Arguments:
        input: one-hot encoded prediction BxCxHxW, output from sigmoid (softmax)
        target: ground-truth
    Output:
        Dice score
    adapted from: https://github.com/pytorch/pytorch/issues/1249
    '''
    smooth = 1. # Additive/ Laplace smoothing
    iflat = input.flatten()
    tflat = target.flatten()
    intersection = np.sum(iflat * tflat)

    return ((2. * intersection + smooth) /
              (np.sum(iflat) + np.sum(tflat) + smooth))

def sensitivity_score(input, target):
    ''' Calculate Sensitivity score
    Arguments:
        input: one-hot encoded prediction BxCxHxW, output from sigmoid (softmax)
        target: ground-truth
    Output:
        sensitivity
    '''
    eps = 1.
    iflat = input.flatten()
    tflat = target.flatten()
    TP = np.sum(np.logical_and(iflat == 1, tflat == 1)) #true positive
    FN = np.sum(np.logical_and(iflat == 0, tflat == 1)) #false negative
    return (TP+eps)/(TP+FN+eps)

def specificity_score(input, target):
    ''' Calculate Specificity score
    Arguments:
        input: one-hot encoded prediction BxCxHxW, output from sigmoid (softmax)
        target: ground-truth
    Output:
        specificity
    '''
    eps = 1.
    iflat = input.flatten()
    tflat = target.flatten()
    TN = np.sum(np.logical_and(iflat == 0, tflat == 0)) #true negative
    FP = np.sum(np.logical_and(iflat == 1, tflat == 0)) #false positive
    return (TN+eps)/(FP+TN+eps)

def test(model, dset, n_classes, start=0, dice_start=0.0, sensitivity_start=0.0,specificity_start=0.0):
    ''' Test a U-Net model, calculate test scores
    Arguments:
        model: U-Net model
        dset: test dataset (from Brats15NumpyDataset)
        n_classes: number of classes
        start: start number of step (if testing is continued from a certain step)
        dice_start: initial dice score (if testing is continued)
        sensitivity_start: initial sensitivity score (if testing is continued)
        specificity_start: initial specificity score (if testing is continued)
    Outputs: 
        test_scores.npz: total mean Dice, sensitivity, and specificity score 
            after 'steps' testing steps
        all_scores.json: dict including step, Dice, sensitivity, and specificity score
    '''
    
    # switch to evaluate mode
    model.eval()

    dloader = DataLoader(dset, batch_size=1, shuffle=False)
    dice = dice_start*(start+1)
    sensitivity = sensitivity_start*(start+1)
    specificity = specificity_start*(start+1)

    all_scores = {'step': [],'dice': [], 'sensitivity': [], 'specificity': []} 
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
            sensitivity += TPR
            TNR = specificity_score(y_pred_1h,y_1h)
            specificity += TNR
            all_scores['step'].append(ii)
            all_scores['dice'].append(float(dice_sc))
            all_scores['sensitivity'].append(TPR)
            all_scores['specificity'].append(TNR)
            if ii%100==0 or ii<10:
                print('step {}: dice={}, sens.={:.3f}, spec.={:.3f}'.format(ii,dice/(ii+1),sensitivity/(ii+1),specificity/(ii+1)))
                
            if ii%500==0:
                np.savez_compressed('./test_scores.npz', dice=dice/(ii+1), steps=ii, sensitivity=sensitivity/(ii+1), specificity=specificity/(ii+1))
                with open('all_scores.json', 'w') as fp:
                    json.dump(all_scores, fp)
            del dice_sc, x, y, y_1h, y_pred_1h, TPR
            
    np.savez_compressed('./test_scores.npz',dice=dice/(ii+1), steps=ii,sensitivity=sensitivity/(ii+1),specificity=specificity/(ii+1))
    with open('all_scores.json', 'w') as fp:
        json.dump(all_scores, fp)
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

##  Train if required:
#    dset_train=Brats15NumpyDataset('./data/brats2015_MR_T2.h5', train=True, train_split=0.8, random_state=-1,
#                 transform=None, preload_data=False, tensor_conversion=False)
#    train(model, dset_train, n_epochs=5, batch_size=2, use_gpu=use_gpu) 
    model.load_state_dict(checkpoint)
    
    #test    
    dset_test=Brats15NumpyDataset(data_file , train=False, train_split=0.8, random_state=-1,
                 transform=None, preload_data=False, tensor_conversion=False)

## Continue from previous testing steps
#    if os.path.isfile('test_scores.npz'):
#        file = np.load('test_scores.npz')
#        dice_start = file['dice']
#        sensitivity_start = file['sensitivity']
#        specificity_start = file['specificity']
#        step = file['steps']
#        test(model, dset_test, 1, start=step+1, dice_start=dice_start, sensitivity_start=sensitivity_start, specificity_start=specificity_start)
#    else:
