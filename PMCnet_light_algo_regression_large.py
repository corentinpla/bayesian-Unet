# # -*- coding: utf-8 -*-
import torch
import math
import numpy as np
from scipy.linalg import sqrtm
from functions import *
import time
import torch.nn.functional as F
import copy
import scipy.io as sio
import sys
import pickle
import os
# from functions import diagevaluate_proposal_multiple_fullCov, Param2vec, evaluate_target_general_Unet, diagweightedcov  


def SL_PMC_Adapt_Cov_new_large(train_loader,N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1, epsilon2,model,optimizer):

    #Initialisation (using ADAM ML solution, maybe not the best idea)
    M_=len(est_ml) # number of unknown parameters # 16
    initial_means= est_ml.repeat(N,1)+torch.randn(N,M_).cuda()
    print("initial means", est_ml.shape)
    #Variance of the proposals 
    Sigma0_small=(torch.ones(M_,1)*(sig_prop**2)).double().cuda()
    Sigma0=Sigma0_small.repeat(1,N) # M*N

    #parameters for the covariance adaptation
    steps = 1
    coef_step = 0.1
    cov_type = 2
    beta0 = 0.5
    increase_beta = 0
    eta0 = 1
    decrease_eta = 1
    PMC_b = True
   

    fraction=(math.sqrt(K))/K

    # variables in which the outputs are stored
    all_proposals=[[] for t in range(T)]
    all_Sigma=[[] for t in range(T)]
    all_samples=[[] for t in range(T)]
    Z_part=[]
    Z_est_evolution=[]
    all_weights=[]
    logf_save = []
    w_sum=[]
    mean_est_part=[]
    m2_part=[]
    Mean_est_evolution=[]
    m2_est_evolution=[]
    sample_Cov=[]
    parents_samples_resampled=[]
    all_samples_resampled_pre=[]
    beta=[]
    eta=[]
    rob_Cov_1=[]
    rob_Cov_2=[]
    ct_vec = []
    Stot=0
    val_max=torch.tensor(-float("Inf"))
    for t in range(T):
        # 0. Update proposals
        if t == 0: # First iteration
            proposals_temp= torch.transpose(initial_means, 0, 1) # Initialized means
            Sigma_temp=Sigma0.cpu()

        else: #Next iterations
            proposals_temp=samples_resampled_adapted.float() # Proposal means are the resampled particles of previous iteration
            Sigma_temp=Sigma_resampled.cpu() #  no sigma adaptation
        # 1. Sampling (propagate proposals)
        all_samples_0 = []
        for n in range(N): # N proposal distributions
            # print("proposal temp", proposals_temp.shape)
            # print("A",proposals_temp[:,n].unsqueeze(-1).repeat(1, K).shape)
            # print("B",torch.randn(M_,K).cuda().shape)
            # print("C",torch.tensor(Sigma_temp[:,n]**(1/2)).unsqueeze(-1).float().cuda().shape)
            a = proposals_temp[:,n].unsqueeze(-1).repeat(1, K)+torch.randn(M_,K).cuda()*torch.tensor(Sigma_temp[:,n]**(1/2)).unsqueeze(-1).float().cuda()
            if n == 0:
                all_samples_0 = a
            else:    
                all_samples_0 = torch.cat((all_samples_0,a),1)
        all_samples[t] = all_samples_0
        temp = diagevaluate_proposal_multiple_fullCov(all_samples[t], proposals_temp, Sigma_temp, N) 
        fp_mixt=np.array(temp.cpu(), dtype=np.float64)
        logP = np.log(fp_mixt+epsilon1)
        # evaluate the Bayes NN target
        sample_number = all_samples[t].shape[1]
        nlogf = []
        for s in range(sample_number):
            _, _, nlogf_temp = evaluate_target_general_Unet(all_samples[t][:,s].double(),tp,model,optimizer)
            print("nlogf_temp",nlogf_temp)
            nlogf.append(nlogf_temp)
        logf = -torch.tensor(nlogf)

        # 2. Weighting
        logf = logf-logf.max()
        logf_np = np.array(logf.detach().numpy())
        w_temp1 = np.exp(logf_np)#+epsilon1
        w_temp2 = np.exp(-logP)
        w = w_temp1*w_temp2
        w = np.nan_to_num(w) 
        if np.sum(w) == 0:
            wn = 1/len(w)*np.ones(len(w))
        else:    
            wn=w/np.sum(w)
        w_sum.append(np.sum(w))# Sum of raw weightsC    


        val_now = torch.max(logf)
        ind_now = torch.argmax(logf)
        if val_now > val_max :
            val_max = val_now

        wn_D = np.repeat(wn.reshape(1,len(wn)),M_,axis=0)

        Stot += w_sum[t]
        all_weights.append(w)

        # 3. Multinomial resampling 
        if lr == 1:
            pos=[]
            for j in range(N):
                proposal_indices = [i for i in range(j*K, (j+1)*K)] # every K elements
                wn_temp = wn[proposal_indices] 
                wn_temp = np.nan_to_num(wn_temp)
                wn_n = (wn_temp+epsilon2)/np.sum(wn_temp+epsilon2)
                wn_n = np.nan_to_num(wn_n)
                pos.append(int(np.random.choice(proposal_indices,1, replace = True, p =wn_n)))


        elif  lr ==2 : # local resampling but every tp.gr_period, we do gr
            if (t+1)%gr_period== 0: # do gr  
                print('do gr')
                pos = np.random.choice([i for i in range(N*K)], N, replace = True, p = wn )
            else : # do lr
                pos=[]
                for j in range(N):
                    proposal_indices = [i for i in range(j*K, (j+1)*K)] # every K elements
                    wn_temp = wn[proposal_indices] 
                    wn_temp = np.nan_to_num(wn_temp)
                    wn_n = (wn_temp+epsilon2)/np.sum(wn_temp+epsilon2)
                    wn_n = np.nan_to_num(wn_n)
                    pos.append(int(np.random.choice(proposal_indices,1, replace = True, p =wn_n)))

        pos=np.array(pos)
        samples_resampled=all_samples[t][:,pos]
        parents_samples_resampled.append((pos//K))
        all_samples_resampled_pre.append(samples_resampled)
        
        # 4 COVARIANCE + SCALE LANGEVIN ADAPTATION
#         4.1 Sample covariance
        sample_Cov_0=[] 
        rob_Cov_1_0=[]   
        rob_Cov_2_0=[]
        all_weights_new = all_weights[t].copy()  
        for n in range(N):
            ind_nn = [i for i in range((n)*K, (n+1)*K)]
            samples_nn = all_samples[t][:,ind_nn]
            weights_nn = all_weights_new[ind_nn]
            weights_nn = np.nan_to_num(weights_nn)
            norm_weights = (weights_nn+epsilon2)/np.sum(weights_nn+epsilon2)
            W_bessel_n = 1 - np.sum(norm_weights**2)
            sample_mean_nn = torch.sum(torch.tensor(norm_weights).cuda().reshape(1,len(norm_weights)).repeat(M_, 1)*samples_nn,1)

            sample_Cov_nn = diagweightedcov(torch.transpose(samples_nn,0,1).double(),norm_weights) # is the sample covariance approximated # it's a symmetric matrix
            norm_weights_cropped = crop_weights(norm_weights,fraction)
            sample_Cov_nn_cropped = diagweightedcov(torch.transpose(samples_nn,0,1).double(),norm_weights_cropped)#  sample_Cov_nn_cropped is (a possibly biased) stable estimator of the covariance matrix
            # set beta
            if increase_beta == 1:
                beta.append(beta0**(-0.5*(t+1)))
            else:
                beta.append(beta0)
            # set eta
            if decrease_eta == 1:
                eta.append(eta0/(t+1))
            else:
                eta.append(eta0)
            if t==0 :
                rob_Cov_1_0.append(sample_Cov_nn)

                rob_Cov_2_0.append((1-eta[t])*sample_Cov_nn + eta[t]*sample_Cov_nn_cropped)
            else :
                parent_index_new = parents_samples_resampled[t][n]

                rob_Cov_1_0.append((1-beta[t])*rob_Cov_1[t-1][parent_index_new] + beta[t]*sample_Cov_nn)
                rob_Cov_2_0.append((1-beta[t])*rob_Cov_2[t-1][parent_index_new] + beta[t]*(1-eta[t])*sample_Cov_nn + beta[t]*eta[t]*sample_Cov_nn_cropped)  

        sample_Cov.append(sample_Cov_0)
        rob_Cov_1.append(rob_Cov_1_0)
        rob_Cov_2.append(rob_Cov_2_0) # ias covariance matrix of the proposal density for next iteration


        # mean adaptation using Langevin
        ct = 0
        Save_A=[]
        samples_resampled_adapted = torch.zeros(M_,N).cuda()
        Sigma_resampled = torch.zeros(M_,N).cuda()
        gradient_save = []
        samples_save = []
        for n in range(N):
            samples_resampled_n = samples_resampled[:,n].reshape(M_,1)
            # use mini-batch
            batch = 0
            for x_train, y_train in train_loader:
                if batch == 0:
                    samples_resampled_n = samples_resampled_n
                elif batch > 0:
                    samples_resampled_n = samples_resampled_adapted_0

                tp_temp = copy.deepcopy(tp)
                tp_temp['x_0'] = x_train
                tp_temp['y'] = y_train
                
                W_g,B_g,l = evaluate_target_general_Unet(samples_resampled_n.double(),tp_temp)
                if cov_type==0 :
                    A_n = sample_Cov[t][n]
                elif cov_type==1:
                    A_n=rob_Cov_1[t][n]
                elif cov_type==2 :    
                    A_n=rob_Cov_2[t][n]    
                flag_A = 1
                if flag_A == 1:
                    if (np.isreal(np.sqrt(A_n.cpu().numpy()))).all() == False:
                        flag_A = 0
                if flag_A == 0:
                    ct = ct+1
                    if t == 0:
                        Save_A.append(Sigma0_small.double().cuda())
                    else:
                        parent_index_new = parents_samples_resampled[t][n]
                        Save_A.append(Past_A[parent_index_new])
                else:
                    Save_A_temp = A_n
                dirn = Param2vec(W_g,B_g,tp_temp)
                dn =  (Save_A_temp.float()*dirn.squeeze(-1)).unsqueeze(-1)
                mysteps = steps
                cts = 0
                f_0=l
                _,_,f_s = evaluate_target_general_Unet((samples_resampled_n - mysteps*dn).double(),tp_temp)
                while f_s > f_0 : #backtracking
                    mysteps = mysteps*coef_step
                    cts = cts + 1
                    _,_,f_s = evaluate_target_general_Unet((samples_resampled_n - mysteps*dn).double(),tp_temp)
                samples_resampled_adapted_0 = (samples_resampled_n - mysteps*dn).double()
                samples_resampled_adapted[:,n] = samples_resampled_adapted_0.squeeze(1)
                Sigma_resampled[:,n] = Save_A_temp#Save_A[n]
                batch += 1
        print(['Iteration ',t,': we discard ',ct,'/',N*(batch+1),' Sigma matrices'])

    return Mean_est_evolution,Z_est_evolution,all_samples, all_weights,all_proposals,all_Sigma,mean_est_part,Z_part,m2_est_evolution,m2_part,samples_resampled_adapted,Sigma_resampled
