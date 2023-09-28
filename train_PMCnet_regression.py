import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
import math
import sys
import torch.utils.data as data
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Optimizer
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal
from sklearn import metrics
import matplotlib.pyplot as plt
from multiprocessing import Pool
import scipy.io as sio
import time
import pandas as pd
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import pickle
from functions import *
from PMCnet_algo import *
from model import Unet
# from golden_search import *
# from run_PMCnet_fixedReg_regression import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import traceback

cuda2 = torch.device('cuda:0')

# folder_name_train = '/workspace/code/PMCnet/data/autoMPG/train_data.mat'
# folder_name_val = '/workspace/code/PMCnet/data/autoMPG/val_data.mat'
# folder_name_test = '/workspace/code/PMCnet/data/autoMPG/test_data.mat'

# y_train = OpenMat(sio.loadmat(folder_name_train)['y_train'])
# x_train1 = OpenMat(sio.loadmat(folder_name_train)['x_train'])
# y_val = OpenMat(sio.loadmat(folder_name_val)['y_val'])
# x_val1 = OpenMat(sio.loadmat(folder_name_val)['x_val'])
# y_test = OpenMat(sio.loadmat(folder_name_test)['y_test'])
# x_test1 = OpenMat(sio.loadmat(folder_name_test)['x_test'])


# x_train = x_train1
# x_val = x_val1
# x_test = x_test1
# scale = StandardScaler()
# x_train[:,:-3] = torch.Tensor(scale.fit_transform(x_train1[:,:-3]))
# x_val[:,:-3] = torch.Tensor(scale.fit_transform(x_val1[:,:-3]))
# x_test[:,:-3] = torch.Tensor(scale.fit_transform(x_test1[:,:-3]))

# y_train = torch.Tensor(y_train)
# y_val = torch.Tensor(y_val)
# y_test = torch.Tensor(y_test)

# print('The size of x_train is ', x_train.shape)
# print('The size of y_train is ', y_train.shape)
# print('The size of x_val is ', x_val.shape)
# print('The size of y_val is ', y_val.shape)
# print('The size of x_test is ', x_test.shape)
# print('The size of y_test is ', y_test.shape)

# batch_size = 10

# train_dataset = TensorDataset(x_train,y_train) # create your datset
# train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True) 


# val_dataset = TensorDataset(x_val,y_val) # create your datset
# val_loader = DataLoader(val_dataset, batch_size = x_val.shape[0], shuffle = True) # not use minibatch


# test_dataset = TensorDataset(x_test,y_test) # create your datset
# test_loader = DataLoader(test_dataset, batch_size =  x_test.shape[0], shuffle = False) # not use minibatch

# # create a dictionary to save options
# hidden_layer1 = 5 # the number of neurons in the hidden layer # L=2
# hidden_layer2 = 0
tp = {}
# tp['L'] = 2
# tp['S'] = 1
# if tp['L'] != 1:
#     tp['M'] = [x_train.shape[1],hidden_layer1,1] # the map
# else:
#     tp['M'] = [x_train.shape[1],1] # the map
tp['prior'] = 'L2' #'no_prior', 'Gaussian_prior', 'Laplace_prior','L2'
tp['x_0'] = x_train
tp['y'] = y_train
tp['regularization_weight'] = 1.480264670576135
if tp['prior'] == 'no_prior':# MLE
    prior_W = 'no_prior'
    prior_b = 'no_prior'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'Gaussian_prior':
    prior_W = isotropic_gauss_prior(mu=0, sigma=2)
    prior_b = isotropic_gauss_prior(mu=0, sigma=2)
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'Laplace_prior':# MAP+L1 regularization
    prior_sig = 0.1
    prior = laplace_prior(mu=0, b=prior_sig)
elif tp['prior'] == 'L2': # L2 regularization
    prior_W = 'L2'
    prior_b = 'L2'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b
elif tp['prior'] == 'L1': # L1 regularization
    prior_W = 'L1'
    prior_b = 'L1'
    tp['prior_W'] = prior_W
    tp['prior_b'] = prior_b       
print('The prior is ',tp['prior'])        
# L = tp['L']
# M = tp['M']
# total_number = 0
# for ll in range(L):
#     total_number += M[ll]*M[ll+1]+M[ll+1]
# tp['dimension'] = total_number  
# tp['activation'] = ['ReLU','None']
# tp['classification'] = 'regression'
dogolden_search = 0
dosave = 0

# some settings
use_cuda = torch.cuda.is_available()
lr = 1e-3
results_dir = '/workspace/code/results_yunshi/autoMPG'  
os.makedirs(results_dir, exist_ok=True) 

# initialize the parameters by the learnt model derived by MLE
# load the parameters 

# folder_name_params = '/workspace/code/PMCnet/params/autoMPG'

# W1 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'W1.mat'))['W1']).cuda()
# b1 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'b1.mat'))['b1']).cuda()
# W2 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'W2.mat'))['W2']).cuda()
# b2 = OpenMat(sio.loadmat(os.path.join(folder_name_params,'b2.mat'))['b2']).cuda()
  

# print('the shape of W1 is ',W1.shape)
# print('the shape of b1 is ',b1.shape)
# if hidden_layer1 != 0:
#     print('the shape of W2 is ',W2.shape)
#     print('the shape of b2 is ',b2.shape)


#parameters for our algorithm
p=time.time()
N = 10 # number of proposals
K = 10  # samples per proposal per iteration
sig_prop = 0.01
lr = 2  #glocal resampling
gr_period=5
tp['regularization_weight'] = 1.480264670576135
epsilon1 = 1e-50
epsilon2 = 1e-50

# W_mu1 = W1
# b_mu1 = b1

# if hidden_layer1 != 0:
#     W_mu2 = W2
#     b_mu2 = b2

# est_ml is the set of all parameters (W,b), stacked in a column #16*1
# if hidden_layer1 == 0:
#     est_ml = torch.cat((torch.transpose(W_mu1,0,1).reshape(M[0]*M[1],1),b_mu1.reshape(M[1],1)),0)
# else:
#     est_ml1 = torch.cat((torch.transpose(W_mu1,0,1).reshape(M[0]*M[1],1),b_mu1.reshape(M[1],1)),0)
#     est_ml2 = torch.cat((torch.transpose(W_mu2,0,1).reshape(M[1]*M[2],1),b_mu2.reshape(M[2],1)),0)
#     est_ml = torch.cat((est_ml1,est_ml2),0)
model= Unet(   
    dim=image_size,
    init_dim=None,
    out_dim=None,
    dim_mults=(1, 2, 4),
    channels=channels,
    with_time_emb=True,
    convnext_mult=2,
).to(DEVICE)

est_ml=[]
for param_tensor in model.state_dict():

    weights = model.state_dict()[param_tensor]
    weights = weights.reshape(16,1)
    est_ml.append(weights)

est_ml=torch.cat(est_ml, axis=0)

        #print(weights.shape)
        #     # Replace this part with the sampling ... 
        #     new_weigths = torch.zeros_like(weights)

        #     # reload the state_dict 
        #     sd[param_tensor] = new_weigths
        #     model.load_state_dict(sd)
logger = get_logger('log_BNN_autoMPG_l2.txt') 

   
T = 50
N_resampled = 200
is_binary = 0
loss = 'MSE'
y_train1 = y_train.detach().numpy()
y_val1 = y_val.detach().numpy()
y_test1 = y_test.detach().numpy()


myprint('T is {}'.format(T),logger)
myprint('regularization_weight is {}'.format(tp['regularization_weight']),logger)
myprint('sig_prop is {}'.format(sig_prop),logger)
myprint('N_resampled is {}'.format(N_resampled),logger)

output_vec = []


    ##This line opens a log file
with open("bug_log_BNN_regression.txt", "w") as log:

    try:
        for i in range(1): 
            myprint('This is simulation {}'.format(i),logger)
            output = SL_PMC_Adapt_Cov_new(N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1,epsilon2)
            output_vec.append(output) 

            path_save_BNN_output  = os.path.join(results_dir,'output_autoMPG_l2_final.txt')             
            with open(path_save_BNN_output, "wb") as fp:   #Pickling
                pickle.dump(output_vec, fp)
        
        print("There is no bug.", file = log)
    except Exception:
        traceback.print_exc(file=log)     



    


