from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from load import Brats15NumpyDataset
import torch 
from torch.utils.data import DataLoader
from u_net import UNet
import json


#%%
n_classes = 1
model = UNet(n_classes)  
checkpoint_file = './unet_dice_e30/training-30-7949.ckpt'
data_file = './data/brats2015_MR_Flair_LGG_r1.h5'

use_gpu = torch.cuda.is_available()
if use_gpu:
    checkpoint = torch.load(checkpoint_file) #gpu
else:
    checkpoint= torch.load(checkpoint_file,map_location=lambda storage, location: storage)
model.load_state_dict(checkpoint)

## Training if required:
#    dset_train=Brats15NumpyDataset('./data/brats2015_MR_T2.h5', train=True, train_split=0.8, random_state=-1,
#                 transform=None, preload_data=False, tensor_conversion=False)
#    
#    train(model, dset_train, n_epochs=5, batch_size=2, use_gpu=use_gpu)

#%% Visualise Training Images

dset_train=Brats15NumpyDataset(data_file, train=True, train_split=0.8, random_state=-1,
                 transform=None, preload_data=False, tensor_conversion=False)
dloader = DataLoader(dset_train, batch_size=1, shuffle=False)
for step, (x, y) in enumerate(dloader):
    if step==84: #select image number

        y = y.numpy().squeeze()
        y[y>0] = 1
        y_pred = np.sign(model(x).detach().numpy().squeeze())
        y_pred[y_pred<0]=0
        x = x.numpy().squeeze()
        break
#%%

plt.figure()
plt.subplot(1,3,1)
plt.imshow(y)
plt.title('train gt')
plt.subplot(1,3,2)
plt.imshow(y_pred)
plt.title('train pred')
plt.subplot(1,3,3)
plt.imshow(x)
plt.title('train img')

plt.figure()
plt.subplot(1,2,1)
plt.imshow(x,cmap='gray')
plt.imshow(y, alpha=0.5)
plt.title('train gt')
plt.subplot(1,2,2)
plt.imshow(x,cmap='gray')
plt.imshow(y_pred,alpha=0.5)
plt.title('train pred')

#%% Visualise Testing Images
dset_test=Brats15NumpyDataset(data_file, train=False, train_split=0.8, random_state=-1,
            transform=None, preload_data=False, tensor_conversion=False)
dloader = DataLoader(dset_test, batch_size=1, shuffle=False)

#%%
for step, (x, y) in enumerate(dloader):
    if step==84: #select image number
        y = y.numpy().squeeze()
        y[y>0] = 1
        y_pred = np.sign(model(x).detach().numpy().squeeze())
        y_pred[y_pred<0]=0
        x = x.numpy().squeeze()
        
        #y_pred = np.argmax(model(x).detach().numpy(), axis=1)
        #y_pred = y_pred.squeeze()

        
        break
    
#%%
plt.figure()
plt.subplot(1,3,1)
plt.imshow(y)
plt.title('test gt')
plt.subplot(1,3,2)
plt.imshow(y_pred)
plt.title('test pred')
plt.subplot(1,3,3)
plt.imshow(x)
plt.title('test img')

plt.figure(figsize=(6,4))
plt.subplot(1,2,1)
plt.imshow(x,cmap='gray')
plt.imshow(y_pred, alpha=0.5)
plt.title('prediction')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(x,cmap='gray')
plt.imshow(y,alpha=0.5)
plt.title('ground truth')
plt.axis('off')
plt.tight_layout()
plt.savefig('test_84.pdf',bbox_inches='tight')


#%% Visualisation of loss functino
with open('unet_dice_e30\loss_history.json') as f:
            history = json.loads(f.read())
#%%
train_step = np.arange(0,6200,200)
loss = history['loss']
del loss[1:10]
#%%
plt.figure()
plt.plot(train_step/264,loss)
plt.ylim([0,1])
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('1-Dice')
plt.xlim([0,23])
#%%
file = np.load('./test_scores.npz')
dice = file['dice']
sensitivity = file['sensitivity']
specificity= file['specificity']
step = file['steps']
print(dice)
print(sensitivity)
print(specificity)




#%% Visualisation of scores as boxplots
with open('all_scores.json') as f:
            all_scores = json.loads(f.read())
#%%         
plt.figure(figsize=(5,3))
plt.boxplot([all_scores['dice'],all_scores['sensitivity'],all_scores['specificity']])
plt.xticks([1,2,3],["Dice","Sensitivity","Specificity"])
plt.tight_layout()

plt.savefig('boxplots.pdf',bbox_inches='tight')
# mean Dice score
print(np.mean(np.array(all_scores['dice'])))
#%%
dice_all = all_scores['dice']
slices = np.arange(0,664,31)
plt.figure()
plt.plot(dice_all)
plt.plot(np.arange(0,664,31),dice_all[0::31],'*')
