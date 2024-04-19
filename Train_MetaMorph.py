from os import PathLike
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os, glob
import json
import subprocess
import sys
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from easydict import EasyDict as edict
import random
import yaml
from Diffeo_losses import NCC, MSE, Grad
from Diffeo_networks import *
from Seg_networks import *
from SitkDataSet import SitkDataset as SData
from uEpdiff import *
import lagomorph as lm 
################### Utility Functions ###################

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None

def load_and_preprocess_data(data_dir, json_file, keyword):
    readfilename = f'{data_dir}/{json_file}.json'
    try:
        with open(readfilename, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f'Error loading JSON data: {e}')
        return None

    outputs = []
    temp_scan = sitk.GetArrayFromImage(sitk.ReadImage(f'{data_dir}/{data[keyword][0]["Source"]}'))
    xDim, yDim, zDim = temp_scan.shape
    return xDim, yDim, zDim

def initialize_network_optimizer(xDim, yDim, zDim, para, dev):
    # Initialize network
    net = DiffeoDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32, 32], [32, 32, 32, 16, 16]],
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    net_seg = SegDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[8, 16], [16, 16, 8, 8]],
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net_seg = net_seg.to(dev)

    # Initialize criterion and optimizer
    if para.model.loss == 'L2':
        criterion = nn.MSELoss()
    elif para.model.loss == 'L1':
        criterion = nn.L1Loss()

    if para.model.optimizer == 'Adam':
        params = list(net.parameters()) + list(net_seg.parameters())
        optimizer = optim.Adam(params, lr=para.solver.lr)
    elif para.model.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=para.solver.lr, momentum=0.9)

    return net, net_seg, criterion, optimizer

def train_network(trainloader, net, net_seg, para, criterion, optimizer, DistType, RegularityType, weight_dist, weight_reg,  reduced_xDim, reduced_yDim, reduced_zDim,  xDim, yDim, zDim, dev, flag):
    # Training loop
    running_loss = 0
    total = 0
    fluid_params = [1.0, 0.1, 0.05]
    lddmm_metirc = lm.FluidMetric(fluid_params)
    print (xDim, yDim, zDim)
    for epoch in range(para.solver.epochs):
        net.train()
        net_seg.train()
        print('epoch:', epoch)

        for j, image_data in enumerate(trainloader):
            src_bch, tar_bch, src_seg_bch, tar_seg_bch = image_data
            b, c, w, h, l = src_bch.shape
            optimizer.zero_grad()

            ''' Getting data '''
            phiinv_bch = torch.zeros(b, w, h, l, 3).to(dev)
            reg_save = torch.zeros(b, w, h, l, 3).to(dev)
            src_bch = src_bch.to(dev).float() 
            tar_bch = tar_bch.to(dev).float() 
            src_seg_bch = src_seg_bch.to(dev).float() 
            tar_seg_bch = tar_seg_bch.to(dev).float()

            '''Computing Union''' 
            if epoch <= para.model.pretrain_epoch:
                union = src_seg_bch + tar_seg_bch 
            else: 
                src_pred = net_seg(src_bch)
                tar_pred = net_seg(tar_bch)
                dice_loss_1 = dice_loss(binarize(src_pred), src_seg_bch )
                dice_loss_2 = dice_loss(binarize(tar_pred), tar_seg_bch)
                dice_loss_total = dice_loss_1 + dice_loss_2
                union = src_pred + tar_pred
             
            binary_map = 1 - union

            '''Mask out lesions and run registration'''
            pred = net(src_bch * binary_map, tar_bch * binary_map, src_bch, binary_map, registration=True, shooting = flag)

            if (flag == "FLDDMM"):
                momentum = pred[1].permute(0, 4, 3, 2, 1)
                print (momentum.shape)
                identity = get_grid2(xDim, dev).permute([0, 4, 3, 2, 1])  
                epd = Epdiff(dev, (reduced_xDim, reduced_yDim, reduced_zDim), (xDim, yDim, zDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

                for b_id in range(b):
                    v_fourier = epd.spatial2fourier(momentum[b_id,...].reshape(w, h , l, 3))
                    velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h , l, 3)  
                    reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                    num_steps = para.solver.Euler_steps
                    v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)  
                    phiinv = displacement.unsqueeze(0) + identity
                    phiinv_bch[b_id,...] = phiinv 
                    reg_save[b_id,...] = reg_temp
                dfm = Torchinterp(src_bch * binary_map,phiinv_bch) 
                Dist = criterion(dfm, tar_bch * binary_map)
                Reg_loss =  reg_save.sum()
                if epoch <= para.model.pretrain_epoch:
                    loss_total =  Dist + weight_reg * Reg_loss
                else:
                    loss_total =  Dist + weight_reg * Reg_loss + dice_loss_total
                print (dfm.shape)
            elif (flag == "SVF"):
                Dist = NCC().loss(pred[0], tar_bch * binary_map)   # Stationary velocity fields to shoot forward when svf = True
                Reg = Grad( penalty= RegularityType)
                Reg_loss  = Reg.loss(pred[1])
                if epoch <= para.model.pretrain_epoch:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss 
                else:
                    loss_total = weight_dist * Dist + weight_reg * Reg_loss + dice_loss_total
            elif (flag == "VecMome"):
                h = lm.expmap(lddmm_metirc, pred[1], num_steps= para.solver.Euler_steps)
                print (h.shape)
                Idef = lm.interp(src_bch* binary_map, h)
                v = lddmm_metirc.sharp(pred[1])
                reg_term = (v*pred[1]).mean()
                
                if epoch <= para.model.pretrain_epoch:
                    loss_total= (1/(para.solver.Sigma*para.solver.Sigma))*NCC().loss(Idef, tar_bch* binary_map) + reg_term
                else:
                    loss_total= (1/(para.solver.Sigma*para.solver.Sigma))*NCC().loss(Idef, tar_bch* binary_map) + reg_term + dice_loss_total
            '''Compute loss'''
            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0
            '''Save and checking results'''
            # velo = pred[1][0].reshape(3, xDim, yDim, zDim).permute(1, 2, 3, 0)
            # velo = velo.detach().cpu().numpy()
            # save_path = f'./check_def/velo_{epoch}_{j}.mhd'
            # sitk.WriteImage(sitk.GetImageFromArray(velo, isVector=True), save_path, False)

            # defim = pred[2][0].reshape(xDim, yDim, zDim).detach().cpu().numpy()
            # save_path = f'./check_def/deform_{epoch}_{j}.mhd'
            # sitk.WriteImage(sitk.GetImageFromArray(defim, isVector=False), save_path, False)

            # tar = tar_bch[0,:,:,:,:].reshape(xDim, yDim, zDim).detach().cpu().numpy()
            # save_path = f'./check_def/tar_{epoch}_{j}.mhd'
            # sitk.WriteImage(sitk.GetImageFromArray(tar, isVector=False), save_path, False)

            # src = src_bch[0,:,:,:,:] .reshape(xDim, yDim, zDim).detach().cpu().numpy()
            # save_path = f'./check_def/src_{epoch}_{j}.mhd'
            # sitk.WriteImage(sitk.GetImageFromArray(src, isVector=False), save_path, False)

            # src_seg_bch = src_seg_bch[0,:,:,:,:] .reshape(xDim, yDim, zDim).detach().cpu().numpy()
            # save_path = f'./check_def/src_seg_{epoch}_{j}.mhd'
            # sitk.WriteImage(sitk.GetImageFromArray(src_seg_bch, isVector=False), save_path, False)

            # tar_seg_bch = tar_seg_bch[0,:,:,:,:] .reshape(xDim, yDim, zDim).detach().cpu().numpy()
            # save_path = f'./check_def/tar_seg_{epoch}_{j}.mhd'
            # sitk.WriteImage(sitk.GetImageFromArray(tar_seg_bch, isVector=False), save_path, False)
        print('Total training loss:', total)

def main():
    dev = get_device()
    para = read_yaml('./parameters.yml')
    data_dir = '.'
    json_file = 'data'
    keyword = 'train'
    xDim, yDim, zDim= load_and_preprocess_data(data_dir, json_file, keyword)
    dataset = SData('./data.json', "train")
    trainloader = DataLoader(dataset, batch_size= para.solver.batch_size, shuffle=False)
    net, net_seg, criterion, optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, dev)
    print (len(trainloader))
    train_network(trainloader, net, net_seg, para, criterion, optimizer, NCC, 'l2', 0.5, 0.5, 16,16,16, xDim, yDim, zDim, dev, "VecMome")

if __name__ == "__main__":
    main()
