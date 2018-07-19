from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, ReLU, MaxPool2d, CrossEntropyLoss
from torch.nn.init import xavier_normal_, normal_
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from load import Brats15NumpyDataset
from u_net import UNet
import json

def dice_loss(input, target):
    '''
    input: one-hot encoded prediction BxCxHxW, output from sigmoid (softmax)
    target: ground-truth
    adapted from: https://github.com/pytorch/pytorch/issues/1249
    '''
    smooth = 1. # Additive/ Laplace smoothing
    iflat = input.view(-1)   
    tflat = target.view(-1)  
    intersection = (iflat * tflat).sum()
    return 1.0 - (((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth)))
    

def train(model, dset, n_epochs=10, batch_size=2, use_gpu=True, load_checkpoint=False, ckpt_file=None):
    '''
    outputs: 
        loss_history.json
        training.pt
        training-{}-{}.ckpt
    '''
    dloader = DataLoader(dset, batch_size=batch_size, shuffle=True)
    if use_gpu:
        model.cuda()
    
    
    #Loss = nn.CrossEntropyLoss() #use if n_classes > 2
    
    Optimizer = torch.optim.Adam(model.parameters()) 
    history = {'train_step': [],'loss': []} 
    start_epoch = 0
    start_e_step = 0
    
    if load_checkpoint:
        if use_gpu:
            checkpoint = torch.load(ckpt_file) #gpu
        else:
            checkpoint= torch.load(ckpt_file,map_location=lambda storage, location: storage)   
    #        model.load_state_dict(checkpoint)
        model.load_state_dict(checkpoint['state_dict'])
        Optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']-1
        start_e_step = (checkpoint['step']+1)%len(dloader)
        if start_e_step == 0:
            start_epoch += 1
        
        print(len(dloader))
        print("=> loaded checkpoint (epoch {} step {})".format(checkpoint['epoch'],checkpoint['step']))
        with open('loss_history.json') as f:
            history = json.loads(f.read())
        
    
    for e in range(start_epoch,n_epochs):
        print('epoch step: {}'.format(e))
        for e_step, (x, y) in enumerate(dloader):
            if e_step >= start_e_step:
                train_step = e_step + len(dloader)*e
                #print(x.size())
                #x = x.numpy()
                #y = y.numpy().astype(np.int32)
    
                #if x.ndim == 2:
                #    x = x[None, :]
                #x = torch.from_numpy(x)
                #y = torch.from_numpy(y)
                y = torch.clamp(y, max=1)
                y = y.float()
                # y= y.long()
                    
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
                #if n_classes > 1:
                    #pred_probs = F.softmax(prediction,dim=1)
                pred_probs = F.sigmoid(prediction)
                #pred_probs = F.softmax(prediction,dim=1)
                loss = dice_loss(pred_probs, y)
                #print('acc')
    		            #acc = torch.mean(torch.eq(torch.argmax(prediction, dim=-1),y).float())
                Optimizer.zero_grad()
                # Backward
                #print('backward')
                loss.backward()              
                # Update
                Optimizer.step()
                del prediction, pred_probs
                if train_step % 200 == 0 or train_step<10:
    #                print('{}: Batch-Accuracy = {}, Loss = {}'\
    #                          .format(train_step, float(acc), float(loss)))
                    print('{}: Loss/ 1-Dice = {}'.format(train_step, float(loss)))
                    history['train_step'].append(train_step)
                    history['loss'].append(float(loss))
                    
                    
                if train_step % 2000 == 0:
                    checkpoint = {
                        'epoch': e + 1,
                        'step': train_step,
                        'state_dict': model.state_dict(),
                        'optimizer' : Optimizer.state_dict(),
                    }
                    torch.save(checkpoint, 'training.pt')
                    with open('loss_history.json', 'w') as fp:
                        json.dump(history, fp)
                    
            
        checkpoint = {
                    'epoch': e + 1,
                    'step': train_step,
                    'state_dict': model.state_dict(),
                    'optimizer' : Optimizer.state_dict(),
                }
        torch.save(model.state_dict(), 'training-{}-{}.pt'.format(e+1,train_step))
        
if __name__ == '__main__':

    n_classes = 1
    use_gpu = torch.cuda.is_available()
    #if 'model' not in locals(): # only reload if model doesn't exist
    model = UNet(n_classes)  
    
    dset_train=Brats15NumpyDataset('./data/brats2015_MR_Flair_LGG_r1.h5', train=True, train_split=0.8, random_state=-1,
                     transform=None, preload_data=False, tensor_conversion=False)
    #train(model, dset_train, n_epochs=10, batch_size=2, use_gpu=use_gpu, load_checkpoint=True, ckpt_file='training.pt') 
    train(model, dset_train, n_epochs=5, batch_size=2, use_gpu=use_gpu, load_checkpoint=True, ckpt_file='training.pt') 
        