from os import PathLike
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import os
import glob
import json
import sys
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from easydict import EasyDict as edict
import random
import yaml
# Custom imports
from Diffeo_losses import NCC, MSE, Grad, Dice
from Diffeo_networks import *
from Seg_networks import *
from SitkDataSet import SitkDataset as SData
from uEpdiff import *
import lagomorph as lm

################### Utility Functions ###################

# Function to determine device (CPU or GPU)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

# Function to read YAML configuration file
def read_yaml(path):
    try:
        with open(path, 'r') as f:
            file = edict(yaml.load(f, Loader=yaml.FullLoader))
        return file
    except:
        print('NO FILE READ!')
        return None

# Function to load and preprocess data
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

# Function to initialize network and optimizer
def initialize_network_optimizer(xDim, yDim, zDim, para, dev):
    # Initialize Diffeo network
    net = DiffeoDense(inshape=(xDim, yDim, zDim),
                      nb_unet_features=[[16, 32, 32], [32, 32, 32, 16, 16]],
                      nb_unet_conv_per_level=1,
                      int_steps=7,
                      int_downsize=2,
                      src_feats=1,
                      trg_feats=1,
                      unet_half_res=True)
    net = net.to(dev)

    # Initialize Segmentation network  Only source channel is inluded for unet segmentation
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

# Function to train the network
def train_network(trainloader, net, net_seg, para, criterion, optimizer, DistType, RegularityType, weight_dist, weight_reg,  reduced_xDim, reduced_yDim, reduced_zDim,  xDim, yDim, zDim, dev, flag):
    # Training loop
    running_loss = 0
    total = 0
    fluid_params = [1.0, 0.1, 0.05]
    lddmm_metirc = lm.FluidMetric(fluid_params)
    print(xDim, yDim, zDim)
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
            union = torch.logical_or(src_seg_bch.bool(), tar_seg_bch.bool()).float() 
            print ("pretrain registratio using gt")
            binary_map = 1 - union
            '''Mask out lesions and run registration'''
            pred = net(src_bch * binary_map, tar_bch * binary_map, src_seg_bch, binary_map, registration=True, shooting=flag)

            
            if (flag == "FLDDMM"):  # Fourier LDDMM to perform geodesic shooting
                momentum = pred[2].permute(0, 4, 3, 2, 1)
                identity = get_grid2(xDim, dev).permute([0, 4, 3, 2, 1])
                epd = Epdiff(dev, (reduced_xDim, reduced_yDim, reduced_zDim), (xDim, yDim, zDim), para.solver.Alpha, para.solver.Gamma, para.solver.Lpow)

                for b_id in range(b):
                    v_fourier = epd.spatial2fourier(momentum[b_id, ...].reshape(w, h, l, 3))
                    velocity = epd.fourier2spatial(epd.Kcoeff * v_fourier).reshape(w, h, l, 3)
                    reg_temp = epd.fourier2spatial(epd.Lcoeff * v_fourier * v_fourier)
                    num_steps = para.solver.Euler_steps
                    v_seq, displacement = epd.forward_shooting_v_and_phiinv(velocity, num_steps)
                    phiinv = displacement.unsqueeze(0) + identity
                    phiinv_bch[b_id, ...] = phiinv
                    reg_save[b_id, ...] = reg_temp
                dfm = Torchinterp(src_bch * binary_map, phiinv_bch)
                Dist = criterion(dfm, tar_bch * binary_map)
                Reg_loss = reg_save.sum()
                loss_total = Dist + weight_reg * Reg_loss
            elif (flag == "SVF"):             # Stationary velocity fields to shoot forward  
                Dist = NCC().loss(pred[0] * binary_map, tar_bch * binary_map)  
                Reg = Grad(penalty=RegularityType)
                Reg_loss = Reg.loss(pred[2])
                loss_total = weight_dist * Dist + weight_reg * Reg_loss
            elif (flag == "VecMome"):             # A spatial version of LDDMM on CUDA to perform geodesic shooting 
                h = lm.expmap(lddmm_metirc, pred[2], num_steps=para.solver.Euler_steps)
                Idef = lm.interp(src_bch * binary_map, h)
                ''' Generated augumented image and its label'''
                v = lddmm_metirc.sharp(pred[1])
                reg_term = (v * pred[1]).mean()
                loss_total = (1 / (para.solver.Sigma * para.solver.Sigma)) * NCC().loss(Idef, tar_bch * binary_map) + reg_term


            '''Compute loss'''
            loss_total.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss_total.item()
            total += running_loss
            running_loss = 0.0
        print('Total training loss:', total)

# Main function
def main():
    # Get the device (CPU or GPU)
    dev = get_device()

    # Read parameters from YAML configuration file
    para = read_yaml('./parameters.yml')

    # Load and preprocess data
    data_dir = '.'
    json_file = 'data'
    keyword = 'train'
    xDim, yDim, zDim = load_and_preprocess_data(data_dir, json_file, keyword)

    # Create dataset and dataloader
    dataset = SData('./data.json', "train")
    trainloader = DataLoader(dataset, batch_size=para.solver.batch_size, shuffle=False)

    # Initialize network and optimizer
    net, net_seg, criterion, optimizer = initialize_network_optimizer(xDim, yDim, zDim, para, dev)

    # Train the network
    train_network(trainloader, net, net_seg, para, criterion, optimizer, NCC, 'l2', 0.5, 0.5, 16, 16, 16, xDim, yDim, zDim, dev, "SVF")

# Entry point of the script
if __name__ == "__main__":
    main()
