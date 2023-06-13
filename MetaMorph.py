from __future__ import division, print_function
from typing import Dict, SupportsRound, Tuple, Any
from os import PathLike
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch,gc
from torch.autograd import grad
from torch.autograd import Variable
import torch.fft ############### Pytorch >= 1.8.0
import torch.nn.functional as F
import SimpleITK as sitk
import os, glob
import json
import subprocess
import sys
from PIL import Image
import torch.nn.functional as nnf
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
import random
import yaml
from Tools import *
from uEpdiff import *

if torch.cuda.is_available():
    dev = "cuda"
else:
    dev = "cpu"

################Parameter Loading#######################
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None
para = read_yaml('./parameters.yml')

xDim = para.data.x 
yDim = para.data.y
zDim = para.data.z

def loss_Reg(y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        dy = dy * dy
        dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy) 
        grad = d / 2.0
        return grad

##################Data Loading##########################
readfilename = './Axial_val/val' + '.json'
datapath = './Axial_val/'
data = json.load(open(readfilename, 'r'))
outputs = []
keyword = 'val'
# outputs = np.array(outputs)

for i in range (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['source']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src).reshape (1, xDim, yDim)
    filename_tar = datapath + data[keyword][i]['target']
    itkimage_tar = sitk.ReadImage(filename_tar)
    target_scan = sitk.GetArrayFromImage(itkimage_tar).reshape (1, xDim, yDim)

    filename_src_s = datapath + data[keyword][i]['src_seg']
    itkimage_src_s = sitk.ReadImage(filename_src_s)
    source_seg = sitk.GetArrayFromImage(itkimage_src_s).reshape (1, xDim, yDim)

    filename_tar_s = datapath + data[keyword][i]['tar_seg']
    itkimage_tar_s = sitk.ReadImage(filename_tar_s)
    target_seg = sitk.GetArrayFromImage(itkimage_tar_s).reshape (1, xDim, yDim)


    pair = np.concatenate((source_scan, source_seg, target_scan, target_seg), axis=0)
    outputs.append(pair)

train = torch.FloatTensor(outputs)
print (train.shape)




readfilename = './Axial_val/val' + '.json'
datapath = './Axial_val/'
data = json.load(open(readfilename, 'r'))
outputs = []
keyword = 'val'
# outputs = np.array(outputs)

for i in range (0,len(data[keyword])):
    filename_src = datapath + data[keyword][i]['source']
    itkimage_src = sitk.ReadImage(filename_src)
    source_scan = sitk.GetArrayFromImage(itkimage_src).reshape (1, xDim, yDim)
    filename_tar = datapath + data[keyword][i]['target']
    itkimage_tar = sitk.ReadImage(filename_tar)
    target_scan = sitk.GetArrayFromImage(itkimage_tar).reshape (1, xDim, yDim)

    filename_src_s = datapath + data[keyword][i]['src_seg']
    itkimage_src_s = sitk.ReadImage(filename_src_s)
    source_seg = sitk.GetArrayFromImage(itkimage_src_s).reshape (1, xDim, yDim)

    filename_tar_s = datapath + data[keyword][i]['tar_seg']
    itkimage_tar_s = sitk.ReadImage(filename_tar_s)
    target_seg = sitk.GetArrayFromImage(itkimage_tar_s).reshape (1, xDim, yDim)

    pair = np.concatenate((source_scan, source_seg, target_scan, target_seg), axis=0)
    outputs.append(pair)

val = torch.FloatTensor(outputs)
print (val.shape)

'''Check initilization'''
from losses import NCC, MSE, Grad, MutualInformation
#################Network optimization########################
from networks import DiffeoDense  
from networks_seg import SegDense

net = DiffeoDense(inshape = (xDim,yDim),
				 nb_unet_features= [[16, 32],[ 32, 32, 16, 16]],
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res= True)
net = net.to(dev)
net_seg = SegDense(inshape = (xDim,yDim),
				 nb_unet_features= [[16, 32],[ 32, 32, 16, 16]],
                 nb_unet_conv_per_level=1,
                 int_steps=7,
                 int_downsize=2,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res= True)
net_seg = net_seg.to(dev)
# print (net)

trainloader = torch.utils.data.DataLoader(train, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)
valloader = torch.utils.data.DataLoader(val, batch_size = para.solver.batch_size, shuffle=True, num_workers=1)

running_loss = 0 
running_loss_val = 0
template_loss = 0
printfreq = 1
sigma = 0.02
repara_trick = 0.0
loss_array = torch.FloatTensor(para.solver.epochs,1).fill_(0)
loss_array_val = torch.FloatTensor(para.solver.epochs,1).fill_(0)
deform_size = [1, xDim, yDim]

if(para.model.loss == 'L2'):
    criterion = nn.MSELoss()
elif (para.model.loss == 'L1'):
    criterion = nn.L1Loss()
if(para.model.optimizer == 'Adam'):
    optimizer = optim.Adam(net.parameters(), lr= para.solver.lr)
elif (para.model.optimizer == 'SGD'):
    optimizer = optim.SGD(net.parameters(), lr= para.solver.lr, momentum=0.9)
if (para.model.scheduler == 'CosAn'):
    scheduler = CosineAnnealingLR(optimizer, T_max=len(valloader), eta_min=0)

optimizer_template = optim.Adam(net.parameters(), lr= para.solver.lr)
scheduler_template = CosineAnnealingLR(optimizer_template, T_max=len(valloader), eta_min=0)
from RMI import RMILoss
loss = RMILoss(radius=5, downsampling_method='max', stride=3, with_logits=True) #Setting logits to be "True"
# ##################Training###################################
for epoch in range(para.solver.epochs):
    total= 0; 
    total_val = 0; 
    total_template = 0; 
    net.train()
    print('epoch:', epoch)
    for j, image_data in enumerate(trainloader):
        inputs = image_data.to(dev)
        b, c, w, h = inputs.shape
        optimizer.zero_grad()
        src_bch = inputs[:,0,...].reshape(b,1,w,h)
        tar_bch = inputs[:,2,...].reshape(b,1,w,h)
        src_seg_bch = inputs[:,1,...].reshape(b,1,w,h)
        tar_seg_bch = inputs[:,3,...].reshape(b,1,w,h)
        union = src_seg_bch + tar_seg_bch
        binary_map = 1 - union
        
        pred = net(src_bch*binary_map, tar_bch*binary_map, src_bch, registration = True)   
        #mi_loss = criterion(pred[0], tar_bch*binary_map) 
        mi_loss = loss (pred[0], tar_bch*binary_map)
        loss2 = loss_Reg(pred[1])
        loss_total = 1*mi_loss + 2*loss2
        loss_total.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss_total.item()
        # print('[%d, %5d] loss: %.3f' %
        #     (epoch + 1, i + 1, running_loss ))
        total += running_loss
        running_loss = 0.0

        # velo = pred[1][0].reshape(2, xDim, yDim).permute(1, 2, 0)
        # velo = velo.detach().cpu().numpy()
        # save_path = './check_def_tumor/velo' + str(epoch) + '.mhd'
        # sitk.WriteImage(sitk.GetImageFromArray(velo, isVector=False), save_path,False)

        # defim = pred[0][0].reshape(xDim, yDim).detach().cpu().numpy()
        # save_path = './check_def_tumor/defim' + str(epoch) + '.mhd'
        # sitk.WriteImage(sitk.GetImageFromArray(defim, isVector=False), save_path,False)



        # tar = inputs[0,2,...].reshape(xDim, yDim).detach().cpu().numpy()
        # save_path = './check_def_tumor/tar' + str(epoch) + '.mhd'
        # sitk.WriteImage(sitk.GetImageFromArray(tar, isVector=False), save_path,False) 

        # src = inputs[0,0,...].reshape(xDim, yDim).detach().cpu().numpy()
        # save_path = './check_def_tumor/src' + str(epoch) + '.mhd'
        # sitk.WriteImage(sitk.GetImageFromArray(src, isVector=False), save_path,False) 

    for j, image_data in enumerate(valloader):
        inputs = image_data.to(dev)
        b, c, w, h = inputs.shape
        optimizer.zero_grad()
        src_bch = inputs[:,0,...].reshape(b,1,w,h)
        tar_bch = inputs[:,2,...].reshape(b,1,w,h)
        src_seg_bch = inputs[:,1,...].reshape(b,1,w,h)
        tar_seg_bch = inputs[:,3,...].reshape(b,1,w,h)
        union = src_seg_bch + tar_seg_bch
        binary_map = 1 - union
        
        pred = net(src_bch*binary_map, tar_bch*binary_map, src_bch, registration = True)   
       

        velo = pred[1][0].reshape(2, xDim, yDim).permute(1, 2, 0)
        velo = velo.detach().cpu().numpy()
        save_path = './check_def_tumor/velo' + str(epoch) + '.mhd'
        sitk.WriteImage(sitk.GetImageFromArray(velo, isVector=False), save_path,False)

        defim = pred[2][0].reshape(xDim, yDim).detach().cpu().numpy()
        save_path = './check_def_tumor/defim' + str(epoch) + '.mhd'
        sitk.WriteImage(sitk.GetImageFromArray(defim, isVector=False), save_path,False)



        tar = inputs[0,2,...].reshape(xDim, yDim).detach().cpu().numpy()
        save_path = './check_def_tumor/tar' + str(epoch) + '.mhd'
        sitk.WriteImage(sitk.GetImageFromArray(tar, isVector=False), save_path,False) 

        src = inputs[0,0,...].reshape(xDim, yDim).detach().cpu().numpy()
        save_path = './check_def_tumor/src' + str(epoch) + '.mhd'
        sitk.WriteImage(sitk.GetImageFromArray(src, isVector=False), save_path,False) 
        
    print ('total training loss:', total)






       
    
 
        


