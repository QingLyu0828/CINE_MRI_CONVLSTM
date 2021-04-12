# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 23:43:46 2020

@author: Qing
"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from Code.ConvLSTM import ConvLSTM
import numpy as np
import h5py
import math
import os
import scipy.io as sio
import pytorch_ssim
#%%
BATCH_SIZE = 16 # Batch size
TEST_BATCH_SIZE = 1
CRITIC_ITERS = 1
MAX_EPOCH = 50
device = torch.device('cuda:0')
DIREC = ''
coef_A = 1.
coef_B = 0.
coef_C = 10.
WEIGHT = [0.25, 0.25, 0.25, 0.25]

DATAPATH = ''
GlobalPATH = ''
#%%
class traindata(data.Dataset):
    def __init__(self):
        path_full = DATAPATH + 'TR.hdf5'  # data size (the_number_of_slices,the_number_of_phases,height,width)
        f = h5py.File(path_full,'r')
        load_data = f['data']
        lr = load_data
        path_full = DATAPATH + 'GT.hdf5'
        f = h5py.File(path_full,'r')
        load_data = f['data']
        hr = load_data
        self.c,self.s,self.h,self.w = lr.shape
                
#        print(lr.shape)
        lr = np.reshape(lr,(self.c*self.s,self.h,self.w)) 
        hr = np.reshape(hr,(self.c*self.s,self.h,self.w))
        
        self.x = lr
        self.y = hr
        self.len = hr.shape[0]
        
    def __getitem__(self, index):
        x = np.zeros((7,self.h,self.w))
        x_inv = np.zeros((7,self.h,self.w))
        y = np.zeros((7,self.h,self.w))
        
        n = index % 30
        if n < 3:         # selecting adjacent 7 phases
            x[0:3-n,:,:] = self.x[index+27:index-n+30,:,:]
            y[0:3-n,:,:] = self.y[index+27:index-n+30,:,:]
            x[3-n:7,:,:] = self.x[index-n:index+4,:,:]
            y[3-n:7,:,:] = self.y[index-n:index+4,:,:]
        elif n > 26:
            x[0:33-n,:,:] = self.x[index-3:index-n+30,:,:]
            y[0:33-n,:,:] = self.y[index-3:index-n+30,:,:]
            x[33-n:7,:,:] = self.x[index-n:index-26,:,:]
            y[33-n:7,:,:] = self.y[index-n:index-26,:,:]
        else:
            x = self.x[index-3:index+4,:,:]
            y = self.y[index-3:index+4,:,:]
               
        for i in range(7):
            x_inv[i] = x[6-i]
            
        x = torch.from_numpy(x)
        x_inv = torch.from_numpy(x_inv)
        y = torch.from_numpy(y)
        
        x = x.type(torch.FloatTensor)
        x_inv = x_inv.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        
        return x, x_inv, y
    
    def __len__(self):
        return self.len
        
class validdata(data.Dataset):
    def __init__(self):
        path_full = DATAPATH + 'VA.hdf5'
        f = h5py.File(path_full,'r')
        load_data = f['data']
        lr = load_data
        path_full = DATAPATH + 'VAGT.hdf5'
        f = h5py.File(path_full,'r')
        load_data = f['data']
        hr = load_data
        self.c,self.s,self.h,self.w = lr.shape
                
#        print(lr.shape)
        lr = np.reshape(lr,(self.c*self.s,self.w,self.h))
        hr = np.reshape(hr,(self.c*self.s,self.w,self.h))
        
        self.x = lr
        self.y = hr
        self.len = hr.shape[0]
        
    def __getitem__(self, index):
        x = np.zeros((7,self.h,self.w))
        x_inv = np.zeros((7,self.h,self.w))
        y = np.zeros((7,self.h,self.w))
        
        n = index % 30
        if n < 3:
            x[0:3-n,:,:] = self.x[index+27:index-n+30,:,:]
            y[0:3-n,:,:] = self.y[index+27:index-n+30,:,:]
            x[3-n:7,:,:] = self.x[index-n:index+4,:,:]
            y[3-n:7,:,:] = self.y[index-n:index+4,:,:]
        elif n > 26:
            x[0:33-n,:,:] = self.x[index-3:index-n+30,:,:]
            y[0:33-n,:,:] = self.y[index-3:index-n+30,:,:]
            x[33-n:7,:,:] = self.x[index-n:index-26,:,:]
            y[33-n:7,:,:] = self.y[index-n:index-26,:,:]
        else:
            x = self.x[index-3:index+4,:,:]
            y = self.y[index-3:index+4,:,:]
               
        for i in range(7):
            x_inv[i] = x[6-i]
            
        x = torch.from_numpy(x)
        x_inv = torch.from_numpy(x_inv)
        y = torch.from_numpy(y)
        
        x = x.type(torch.FloatTensor)
        x_inv = x_inv.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        
        return x, x_inv, y
    
    def __len__(self):
        return self.len
        
class testdata(data.Dataset):
    def __init__(self):
        path_full = DATAPATH + 'TE.hdf5'
        f = h5py.File(path_full,'r')
        load_data = f['data']
        lr = load_data
        path_full = DATAPATH + 'TEGT.hdf5'
        f = h5py.File(path_full,'r')
        load_data = f['data']
        hr = load_data  
              
        self.c,self.s,self.h,self.w = lr.shape
                
        self.x = lr
        self.y = hr
        self.len = lr.shape[0]
        
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)
        
        return x, y
    
    def __len__(self):
        return self.len

#%%
# encoder of the encoder-decoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ########################## T #################################
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(1, 32, 5, padding=2, bias=False)
        self.relu1_2 = nn.ReLU()
        self.conv1_3 = nn.Conv2d(1, 32, 7, padding=3, bias=False)
        self.relu1_3 = nn.ReLU()
        self.conv1_4 = nn.Conv2d(32*3, 32, 1, padding=0, bias=False)        
        self.relu1_4 = nn.ReLU()
        self.conv1_5 = nn.Conv2d(32*3, 32, 3, padding=1, bias=False)        
        self.relu1_5 = nn.ReLU()

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(32, 64, 5, padding=2, bias=False)
        self.relu2_2 = nn.ReLU()
        self.conv2_3 = nn.Conv2d(32, 64, 7, padding=3, bias=False)
        self.relu2_3 = nn.ReLU()
        self.conv2_4 = nn.Conv2d(64*3, 64, 1, padding=0, bias=False)        
        self.relu2_4 = nn.ReLU()
        self.conv2_5 = nn.Conv2d(64*3, 64, 3, padding=1, bias=False)        
        self.relu2_5 = nn.ReLU()

        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1, bias=False)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(64, 128, 5, padding=2, bias=False)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(64, 128, 7, padding=3, bias=False)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv2d(128*3, 128, 1, padding=0, bias=False)        
        self.relu3_4 = nn.ReLU()
        self.conv3_5 = nn.Conv2d(128*3, 128, 3, padding=1, bias=False)        
        self.relu3_5 = nn.ReLU()
        
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1, bias=False)        
        self.relu4_1 = nn.ReLU()
        
    def forward(self, input1, fside1, fside2, fside3, bside1, bside2, bside3):
        sides = []
        out1_1 = self.conv1_1(input1)
        out1_1 = self.relu1_1(out1_1)
        out1_2 = self.conv1_2(input1)
        out1_2 = self.relu1_2(out1_2)
        out1_3 = self.conv1_3(input1)
        out1_3 = self.relu1_3(out1_3)
        out1_4 = torch.cat((out1_1, out1_2, out1_3), 1)
        out1_4 = self.conv1_4(out1_4)
        out1_4 = self.relu1_4(out1_4)        
        out1_5 = torch.cat((out1_4, fside1, bside1), 1)
        out1_5 = self.conv1_5(out1_5)
        out1_5 = self.relu1_5(out1_5)
        sides.append(out1_5)
        
        out2_1 = self.conv2_1(out1_5)
        out2_1 = self.relu2_1(out2_1)
        out2_2 = self.conv2_2(out1_5)
        out2_2 = self.relu2_2(out2_2)
        out2_3 = self.conv2_3(out1_5)
        out2_3 = self.relu2_3(out2_3)
        out2_4 = torch.cat((out2_1, out2_2, out2_3), 1)
        out2_4 = self.conv2_4(out2_4)
        out2_4 = self.relu2_4(out2_4)        
        out2_5 = torch.cat((out2_4, fside2, bside2), 1)
        out2_5 = self.conv2_5(out2_5)
        out2_5 = self.relu2_5(out2_5)
        sides.append(out2_5)
        
        out3_1 = self.conv3_1(out2_5)
        out3_1 = self.relu3_1(out3_1)
        out3_2 = self.conv3_2(out2_5)
        out3_2 = self.relu3_2(out3_2)
        out3_3 = self.conv3_3(out2_5)
        out3_3 = self.relu3_3(out3_3)
        out3_4 = torch.cat((out3_1, out3_2, out3_3), 1)
        out3_4 = self.conv3_4(out3_4)
        out3_4 = self.relu3_4(out3_4)        
        out3_5 = torch.cat((out3_4, fside3, bside3), 1)
        out3_5 = self.conv3_5(out3_5)
        out3_5 = self.relu3_5(out3_5)
        sides.append(out3_5)
        
        out4_1 = self.conv4_1(out3_5)
        out4_1 = self.relu4_1(out4_1)
        sides.append(out4_1)
        
        return sides[::-1]
#%%
# Forward branch of ConvLSTM
class Forward(nn.Module):
    def __init__(self):
        super(Forward, self).__init__()
        ########################## T #################################
        self.conv1_1 = ConvLSTM(1, 32, (3,3), 1, True, True, False)
        self.relu1_1 = nn.ReLU()

        self.conv2_1 = ConvLSTM(32, 64, (3,3), 1, True, True, False)
        self.relu2_1 = nn.ReLU()

        self.conv3_1 = ConvLSTM(64, 128, (3,3), 1, True, True, False)
        self.relu3_1 = nn.ReLU()
        
    def forward(self, input1):
        b,c,h,w = input1.size()
        input1 = input1.view(b,c,1,h,w)
        out1_1= self.conv1_1(input1)
        out1_1 = self.relu1_1(out1_1)

        out2_1 = self.conv2_1(out1_1)
        out2_1 = self.relu2_1(out2_1)
        
        out3_1 = self.conv3_1(out2_1)
        out3_1 = self.relu3_1(out3_1)

        return out1_1, out2_1, out3_1
#%%
# Backward branch of ConvLSTM
class Backward(nn.Module):
    def __init__(self):
        super(Backward, self).__init__()
        ########################## T #################################
        self.conv1_1 = ConvLSTM(1, 32, (3,3), 1, True, True, False)
        self.relu1_1 = nn.ReLU()

        self.conv2_1 = ConvLSTM(32, 64, (3,3), 1, True, True, False)
        self.relu2_1 = nn.ReLU()

        self.conv3_1 = ConvLSTM(64, 128, (3,3), 1, True, True, False)
        self.relu3_1 = nn.ReLU()
        
    def forward(self, input1):
        b,c,h,w = input1.size()
        input1 = input1.view(b,c,1,h,w)
        out1_1 = self.conv1_1(input1)
        out1_1 = self.relu1_1(out1_1)

        out2_1 = self.conv2_1(out1_1)
        out2_1 = self.relu2_1(out2_1)
        
        out3_1 = self.conv3_1(out2_1)
        out3_1 = self.relu3_1(out3_1)
        
        return out1_1, out2_1, out3_1

#%%
# decoder of the encoder-decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        ########################## T #################################
        self.conv4_2 = nn.ConvTranspose2d(256, 128, 3, padding=1, bias=False)        
        self.relu4_2 = nn.ReLU()

        self.conv5_1 = nn.ConvTranspose2d(128*2, 128, 3, padding=1, bias=False)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.ConvTranspose2d(128, 64, 3, padding=1, bias=False)        
        self.relu5_2 = nn.ReLU()

        self.conv6_1 = nn.ConvTranspose2d(64*2, 64, 3, padding=1, bias=False)
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = nn.ConvTranspose2d(64, 32, 3, padding=1, bias=False)        
        self.relu6_2 = nn.ReLU()
        
        self.conv7_1 = nn.ConvTranspose2d(32*2, 32, 3, padding=1, bias=False)
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = nn.ConvTranspose2d(32, 1, 3, padding=1, bias=False)        
        self.relu7_2 = nn.ReLU()
        
    def forward(self, side1):
#        out4_1 = torch.cat((side1[0], side2, side3), 1) 
        out4_2 = self.conv4_2(side1[0])
        out4_2 = self.relu4_2(out4_2)
        
        out5_1 = torch.cat((out4_2, side1[1]), 1)
        out5_1 = self.conv5_1(out5_1)
        out5_1 = self.relu5_1(out5_1)
        out5_2 = self.conv5_2(out5_1)
        out5_2 = self.relu5_2(out5_2)

        out6_1 = torch.cat((out5_2, side1[2]), 1)
        out6_1 = self.conv6_1(out6_1)
        out6_1 = self.relu6_1(out6_1)
        out6_2 = self.conv6_2(out6_1)
        out6_2 = self.relu6_2(out6_2)

        out7_1 = torch.cat((out6_2, side1[3]), 1)
        out7_1 = self.conv7_1(out7_1)
        out7_1 = self.relu7_1(out7_1)
        out7_2 = self.conv7_2(out7_1)
        out7_2 = self.relu7_2(out7_2)
        
        return out7_2
#%%
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1, bias=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(64, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, 3, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=True),
            nn.ReLU(True),
            )
        self.main  = main
        self.fc1   = nn.Linear(5*5*256, 1024)
        self.relu1 = nn.ReLU(True)
        self.fc2   = nn.Linear(1024,1)
        self.relu2 = nn.ReLU(True)

    def forward(self, input):
        out = self.main(input)
        out = out.view(-1, 5*5*256)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        return out
#%%
# for perceptual loss
class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        out = []
        out.append(h_relu1_2)
        out.append(h_relu2_2)
        out.append(h_relu3_3)
        out.append(h_relu4_3)
        return out
#%%   
def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    c,s,h,w = real_data.shape
    alpha = torch.rand(c, s, h, w)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(device) if torch.cuda.is_available() else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) if torch.cuda.is_available() else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
#%%
def concatanate_normalize(x):
    # normalize using imagenet mean and std 
    return torch.cat(((x-0.485)/0.229, (x-0.456)/0.224, (x-0.406)/0.225), 1)

#%%
def train(restore_epoch, continue_train):
    
    net1 = Encoder()
    net2 = Forward()
    net3 = Backward()
    net4 = Decoder()
    netD = Discriminator()
    vgg = Vgg16()
    
    if torch.cuda.device_count() > 1:
        net1 = nn.DataParallel(net1)
        net2 = nn.DataParallel(net2)
        net3 = nn.DataParallel(net3)
        net4 = nn.DataParallel(net4)
        netD = nn.DataParallel(netD)
        vgg = nn.DataParallel(vgg)
        
    net1 = net1.to(device)
    net2 = net2.to(device)
    net3 = net3.to(device)
    net4 = net4.to(device)
    netD = netD.to(device)
    vgg = vgg.to(device)
    
    params = list(net1.parameters()) + list(net2.parameters()) + list(net3.parameters()) + list(net4.parameters())
    optimizerG = optim.Adam(params, lr=1e-4, betas=(0.9, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.9, 0.999))
    
    mse = torch.nn.MSELoss()
    
    # record loss
    adv_ls = []
    per_ls = []
    mse_ls = []
    ssim_ls = []
    G_ls = []
    D_ls = []
    W_ls = []
    perv_ls = []
    msev_ls = []
    ssimv_ls = []
    
    # load pretrained model
    if continue_train:
        checkpoint = torch.load(GlobalPATH + 'Network/' + DIREC + '/model_epoch_' + str(restore_epoch) + '.pkl')
        net1.load_state_dict(checkpoint['Encoder'])
        net2.load_state_dict(checkpoint['Forward'])
        net3.load_state_dict(checkpoint['Backward'])
        net4.load_state_dict(checkpoint['Decoder'])
        netD.load_state_dict(checkpoint['Discriminator'])
        optimizerG.load_state_dict(checkpoint['opG'])
        optimizerD.load_state_dict(checkpoint['opD'])
        net1.eval()
        net2.eval()
        net3.eval()
        net4.eval()
        netD.eval()
        
        # load loss data saved in .mat        
        readmat = sio.loadmat('./Loss/' + DIREC)
        load_adv_loss = readmat['adv']
        load_per_loss = readmat['per']
        load_mse_loss = readmat['mse']
        load_ssim_loss = readmat['ssim']
        load_G_loss = readmat['G']
        load_D_loss = readmat['D']
        load_W_loss = readmat['W']
        load_perv_loss = readmat['perv']
        load_msev_loss = readmat['msev']
        load_ssimv_loss = readmat['ssimv']
        for i in range(restore_epoch):
            adv_ls.append(load_adv_loss[0][i])
            per_ls.append(load_per_loss[0][i])
            mse_ls.append(load_mse_loss[0][i])
            ssim_ls.append(load_ssim_loss[0][i])
            G_ls.append(load_G_loss[0][i])
            D_ls.append(load_D_loss[0][i])
            W_ls.append(load_W_loss[0][i])
            perv_ls.append(load_perv_loss[0][i])
            msev_ls.append(load_msev_loss[0][i])
            ssimv_ls.append(load_ssimv_loss[0][i])
        print('Finish loading loss!')

    if not os.path.exists(GlobalPATH + 'Network/' + DIREC):
        os.makedirs(GlobalPATH + 'Network/' + DIREC)

    for epoch in range(restore_epoch, MAX_EPOCH):
        
        tmp_adv_loss = 0.
        tmp_per_loss = 0.
        tmp_mse_loss = 0.
        tmp_ssim_loss = 0.
        tmp_G_loss = 0.
        tmp_D_loss = 0.
        tmp_W_loss = 0.

        count = 0
        trainset = traindata()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
                
        for i, (x_s, x_l, y) in enumerate(trainloader):
            c,s,h,w = y.shape
            
            optimizerG.zero_grad()
            optimizerD.zero_grad()
            
            x_s_inv = torch.zeros(c,s,h,w)           
            for k in range(s):
                x_s_inv[:,k,:,:] = x_s[:,s-1-k,:,:] # inverse sequence for backward ConvLSTM branch
            
            x_s = x_s.to(device)
            x_s_inv = x_s_inv.to(device)
            x_l = x_l.to(device)
            y = y.to(device)
            
            net1.zero_grad()
            net2.zero_grad()
            net3.zero_grad()
            net4.zero_grad()
            
            ########################
            #   Update Generator   #
            ########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            
            x_s_input = autograd.Variable(x_s)
            x_s_inv_input = autograd.Variable(x_s_inv)
            x_l_input = autograd.Variable(x_l)
            y_input = autograd.Variable(y)
           
            per_loss = 0.
            adv_loss = 0.
            mse_loss = 0.
            ssim_loss = 0.

            f_side1, f_side2, f_side3 = net2(x_s_input)  # forward ConvLSTM branch
            b_side1, b_side2, b_side3  = net3(x_s_inv_input)  # backward ConvLSTM branch
                      
            sides = []
              
            for k in range(s):  # combination of encoder with ConvLSTMs
                tmp_out = net1(x_l_input[:,k,:,:].view(c,1,h,w), f_side1[:,k,:,:,:], 
                               f_side2[:,k,:,:,:], f_side3[:,k,:,:,:], 
                               b_side1[:,s-1-k,:,:,:], b_side2[:,s-1-k,:,:,:], 
                               b_side3[:,s-1-k,:,:,:])
                sides.append(tmp_out)
                            
            for k in range(s):   # decoder prediction results
                out = net4(sides[k])
                D_fake = netD(out)
                adv_loss += -D_fake.mean() / s 
                
                fake_feature = vgg(concatanate_normalize(out))
                real_feature = vgg(concatanate_normalize(y_input[:,k,:,:].view(c,1,h,w)))
            
                for per_num in range(4):  # perceptual loss
                    per_loss += WEIGHT[per_num] * mse(fake_feature[per_num], real_feature[per_num]) / s
                
                mse_loss += mse(out, y_input[:,k,:,:].view(c,1,h,w)) / s  # MSE loss
                
                ssim_loss += (1 - pytorch_ssim.ssim(y_input[:,k,:,:].view(c,1,h,w), out)) / s    # SSIM loss
            
            G_loss = adv_loss + coef_A * per_loss
            
            G_loss.backward()
            optimizerG.step()   
                       
            ############################
            #   Update Discriminator   #
            ############################
            for p in netD.parameters():
                p.requires_grad = True  # to avoid computation
                
            for iter_d in range(CRITIC_ITERS):
                optimizerD.zero_grad()
                
                real_data = autograd.Variable(y)
                    
                D_real = 0.
                D_fake = 0.
                gradient_penalty = 0.
                
                for k in range(s):
                    real_out = netD(real_data[:,k,:,:].view(c,1,h,w))
                    D_real += real_out.mean() / s
                    
                with torch.no_grad():
                    fx_s_input = autograd.Variable(x_s)        
                    fx_s_inv_input = autograd.Variable(x_s_inv)
                    fx_l_input = autograd.Variable(x_l)        
                    
                f_side1, f_side2, f_side3 = net2(fx_s_input)
                b_side1, b_side2, b_side3  = net3(fx_s_inv_input)
                sides = []
                
                for k in range(s):
                    tmp_out = net1(fx_l_input[:,k,:,:].view(c,1,h,w), f_side1[:,k,:,:,:], 
                                   f_side2[:,k,:,:,:], f_side3[:,k,:,:,:], 
                                   b_side1[:,s-1-k,:,:,:], b_side2[:,s-1-k,:,:,:], 
                                   b_side3[:,s-1-k,:,:,:])
                    sides.append(tmp_out)
                            
                for k in range(s):
                    out = net4(sides[k])
                    fake_data = autograd.Variable(out.data)
                    fake_out = netD(fake_data)
                    D_fake += fake_out.mean() / s 
                    
                    gradient_penalty += calc_gradient_penalty(netD, real_data[:,k,:,:].view(c,1,h,w).data, fake_data.data) / s
                
                Wasserstein_D = D_real - D_fake
                D_loss = -Wasserstein_D + coef_C * gradient_penalty
                                      
                D_loss.backward()
                optimizerD.step()      
                
            tmp_adv_loss += adv_loss.item()
            tmp_per_loss += per_loss.item()
            tmp_mse_loss += mse_loss.item()
            tmp_ssim_loss += ssim_loss.item()
            tmp_G_loss += G_loss.item()
            tmp_D_loss += D_loss.item()
            tmp_W_loss += Wasserstein_D.item()
            
            count += 1
            
            print('[Epoch: %d/%d, Batch: %d/%d] G: %.4f, D: %.4f, adv: %.4f, per: %.4f, ssim: %.4f, mse: %.4f, W: %.4f' 
                  % (epoch+1, MAX_EPOCH, i+1, len(trainloader), G_loss.item(), D_loss.item(), adv_loss.item(), 
                     per_loss.item(), ssim_loss.item(), mse_loss.item(), Wasserstein_D.item()))
            assert (D_loss.item() != 10.) # prevent inappropriate parameter initialization
            
        # validation        
        countv = 0
        tmp_perv_loss = 0.
        tmp_msev_loss = 0.
        tmp_ssimv_loss = 0.
        valset = validdata()
        valloader = torch.utils.data.DataLoader(valset, batch_size=TEST_BATCH_SIZE, shuffle=False)
        for i, (x_s, x_l, y) in enumerate(valloader):
            x_s = x_s[:,0:s,:,:]
            x_l = x_l[:,0:s,:,:]
            y = y[:,0:s,:,:]
            c,s,h,w = x_s.shape
            x_s_inv = torch.zeros(c,s,h,w)      
            for k in range(s):
                x_s_inv[:,k,:,:] = x_s[:,s-1-k,:,:]
                
            x_s = x_s.to(device)
            x_s_inv = x_s_inv.to(device)
            x_l = x_l.to(device)
            y = y.to(device)
                
            with torch.no_grad():
                x_s_input = autograd.Variable(x_s)
                x_s_inv_input = autograd.Variable(x_s_inv)
                x_l_input = autograd.Variable(x_l)
                y_input = autograd.Variable(y)
                
            perv_loss = 0.
            ssimv_loss = 0.
            msev_loss = 0.
                
            f_side1, f_side2, f_side3 = net2(x_s_input)
            b_side1, b_side2, b_side3  = net3(x_s_inv_input)
                      
            vsides = []
              
            for k in range(s):
                tmp_out = net1(x_l_input[:,k,:,:].view(c,1,h,w), f_side1[:,k,:,:,:], 
                               f_side2[:,k,:,:,:], f_side3[:,k,:,:,:], 
                               b_side1[:,s-1-k,:,:,:], b_side2[:,s-1-k,:,:,:], 
                               b_side3[:,s-1-k,:,:,:], )
                vsides.append(tmp_out)
                            
            for k in range(s):             
                out = net4(vsides[k])              
                fake_feature = vgg(concatanate_normalize(out))
                real_feature = vgg(concatanate_normalize(y_input[:,k,:,:].view(c,1,h,w)))
                for per_num in range(4):
                    perv_loss += WEIGHT[per_num] * mse(fake_feature[per_num], real_feature[per_num]) / s
                msev_loss += mse(out, y_input[:,k,:,:].view(c,1,h,w)) / s
                ssimv_loss += (1 - pytorch_ssim.ssim(y_input[:,k,:,:].view(c,1,h,w), out)) / s        
            
            tmp_perv_loss += perv_loss.item()
            tmp_msev_loss += msev_loss.item()
            tmp_ssimv_loss += ssimv_loss.item()
            
            countv += 1
                          
        adv_ls.append(tmp_adv_loss/count)   # recording average loss terms in an epoch
        per_ls.append(tmp_per_loss/count)   
        mse_ls.append(tmp_mse_loss/count)   
        ssim_ls.append(tmp_ssim_loss/count)   
        G_ls.append(tmp_G_loss/count)   
        D_ls.append(tmp_D_loss/count)   
        W_ls.append(tmp_W_loss/count)   
        perv_ls.append(tmp_perv_loss/countv)   
        msev_ls.append(tmp_msev_loss/countv)   
        ssimv_ls.append(tmp_ssimv_loss/countv)   
        
        # save all loss terms
        sio.savemat('./Loss/' + DIREC +'.mat', {'G': G_ls, 'D': D_ls, 'adv': adv_ls, 'per': per_ls, 'ssim': ssim_ls, 'mse': mse_ls, 
                                                'W': W_ls, 'perv': perv_ls, 'ssimv': ssimv_ls, 'msev': msev_ls})
        
        # save models
        if (epoch+1) % 5 == 0:
            torch.save({'Forward': net2.state_dict(),'Backward': net3.state_dict(),
                        'Decoder': net4.state_dict(), 'Encoder': net1.state_dict(),'Discriminator': netD.state_dict(),
                        'opG': optimizerG.state_dict(),'opD': optimizerD.state_dict()}, 
                        GlobalPATH + 'Network/' + DIREC + '/model_epoch_'+str(epoch+1)+'.pkl')
    
    print('Finished Training')
#%%   
def test(restore_epoch):
    sr = np.zeros((100, 7, 100, 100))
    
    net1 = Encoder()
    net2 = Forward()
    net3 = Backward()
    net4 = Decoder()
    
    if torch.cuda.device_count() >= 1:
        net1 = nn.DataParallel(net1)
        net2 = nn.DataParallel(net2)
        net3 = nn.DataParallel(net3)
        net4 = nn.DataParallel(net4)
    
    net1 = net1.to(device)
    net2 = net2.to(device)
    net3 = net3.to(device)
    net4 = net4.to(device)
        
    checkpoint = torch.load(GlobalPATH + 'Network/' + DIREC + '/model_epoch_' + str(restore_epoch) + '.pkl')
    net1.load_state_dict(checkpoint['Encoder'])
    net2.load_state_dict(checkpoint['Forward'])
    net3.load_state_dict(checkpoint['Backward'])
    net4.load_state_dict(checkpoint['Decoder'])
    net1.eval()
    net2.eval()
    net3.eval()
    net4.eval()
       
    testset = testdata()
    testloader = torch.utils.data.DataLoader(testset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    for i, (x, y) in enumerate(testloader):
        
        ppart = 1
        x = x[:,ppart*7-7:ppart*7,:,:]
        y = y[:,ppart*7-7:ppart*7,:,:]
        c,s,h,w = x.shape
        x_inv = torch.zeros(c,7,h,w)
        
        for k in range(7):
            x_inv[:,k,:,:] = x[:,6-k,:,:]
            
        x = x.to(device)
        x_inv = x_inv.to(device)
        y = y.to(device)
            
        with torch.no_grad():
            x_input = autograd.Variable(x)
            x_inv_input = autograd.Variable(x_inv)
            y_input = autograd.Variable(y)
            
        f_side1, f_side2, f_side3 = net2(x_input)
        b_side1, b_side2, b_side3  = net3(x_inv_input)
            
        sides = []
        
        for k in range(7):
            tmp_out = net1(x_input[:,k,:,:].view(c,1,h,w), f_side1[:,k,:,:,:], 
                           f_side2[:,k,:,:,:], f_side3[:,k,:,:,:], 
                           b_side1[:,6-k,:,:,:], b_side2[:,6-k,:,:,:], 
                           b_side3[:,6-k,:,:,:], )
            sides.append(tmp_out)
            
        for k in range(7):             
            out = net4(sides[k])
            
            out = out.data.squeeze()                    
            sr[i*TEST_BATCH_SIZE:(i+1)*TEST_BATCH_SIZE,k,:,:] = out.cpu()
                    
    if not os.path.exists(GlobalPATH + 'Result/' + DIREC):
        os.makedirs(GlobalPATH + 'Result/' + DIREC)
    # save prediction results in .hdf5
    path = GlobalPATH + 'Result/' + DIREC + '/' + repr(restore_epoch) + '_'+ str(ppart) + '.hdf5'
    f = h5py.File(path, 'w')
    f.create_dataset('sr', data=sr)
    print('Finish Testing')
#%%

#  Train
restore_epoch = 0
continue_train = False

#  Test
#continue_train = False
#for i in range(5):
#    restore_epoch = (i+1)*10
#    test(restore_epoch)