# # -*- coding: utf-8 -*-
import torch
import math
import numpy as np
from scipy.linalg import sqrtm
from functions import *
import time
import torch.nn.functional as F
import scipy.io as sio
import sys
import pickle
import os


def SL_PMC_Adapt_Cov_new(N,K,T,sig_prop,lr,gr_period,tp,est_ml,epsilon1, epsilon2):

    # initialization
    M=len(est_ml)
    initial_means= est_ml.repeat(1,N)+torch.randn(M,N).cuda() # initial_means M*N 

    #Variance of the proposals 
    Sigma0_small=torch.diag(torch.ones(M)*(sig_prop**2)).cuda().double()
    Sigma0=Sigma0_small.unsqueeze(-1).repeat(1,1,N) # M*M*N
    
    #parameters for the covariance adaptation
    steps = 1
    coef_step = 0.5
    cov_type = 2
    beta0 = 0.5
    increase_beta = 0
    eta0 = 1
    decrease_eta = 1

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
            all_proposals[t]=initial_means # Initialized means
            all_Sigma[t]=Sigma0.cpu()

        else: #Next iterations
            all_proposals[t]=samples_resampled_adapted.float() # Proposal means are the resampled particles of previous iteration
            all_Sigma[t]=Sigma_resampled.cpu() #  no sigma adaptation
        # 1. Sampling (propagate proposals)
        all_samples_0 = []
        for n in range(N): # N proposal distributions
            a = all_proposals[t][:,n].unsqueeze(-1).repeat(1, K)+torch.tensor(sqrtm(all_Sigma[t][:,:,n]).real).float().cuda()@torch.randn(M,K).cuda()
            if n == 0:
                all_samples_0 = a
            else:    
                all_samples_0 = torch.cat((all_samples_0,a),1)      
        all_samples[t] = all_samples_0
        temp = evaluate_proposal_multiple_fullCov(all_samples[t], all_proposals[t], all_Sigma[t], N) 
        fp_mixt=np.array(temp.cpu(), dtype=np.float64)
        logP = np.log(fp_mixt+epsilon1)
        print(logP)
        # evaluate the Bayes NN target
        sample_number = all_samples[t].shape[1]
        nlogf = []
        for s in range(sample_number):
            _,_,nlogf_temp = evaluate_target_general(all_samples[t][:,s].double(),tp)
            nlogf.append(nlogf_temp)
        logf = -torch.tensor(nlogf)

        # 2. Weighting
        logf_np = np.array(logf)
        w = np.exp(logf_np-logP,dtype=np.float64)+epsilon1
        w = np.nan_to_num(w) 
        if np.sum(w) == 0:
            wn = 1/len(w)*np.ones(len(w))
        else:    
            wn=w/np.sum(w)
        w_sum.append(np.sum(w))# Sum of raw weightsC    
        
        Z_part.append(w_sum[t]/(N*K))
        if t == 0: # First iteration
            Z_est_evolution.append(w_sum[t]/(N*K)) # % First Z estimator
        else: # Next iterations
            Z_est_evolution.append((t*Z_est_evolution[t-1] + w_sum[t]/(K*N))/(t+1)) # Recursive Z estimator

        val_now = torch.max(logf)
        ind_now = torch.argmax(logf)
        if val_now > val_max :
            val_max = val_now
            map_estimator = all_samples[t][:,ind_now]

        wn_D = np.repeat(wn.reshape(1,len(wn)),M,axis=0)

        mean_est_part.append(torch.sum(torch.tensor(wn_D).cuda()*all_samples[t],1))

        m2_part.append(torch.sum(torch.tensor(wn_D).cuda()*(all_samples[t]**2),1))

        if t == 0:
            Mean_est_evolution.append(mean_est_part[t])
            m2_est_evolution.append(m2_part[t])
        else :
            Mean_est_evolution.append((Stot* Mean_est_evolution[t-1]+w_sum[t]*mean_est_part[t])/(Stot+w_sum[t]))
            m2_est_evolution.append((Stot* m2_est_evolution[t-1]+w_sum[t]*m2_part[t])/(Stot+w_sum[t]))
        Stot += w_sum[t]
        all_weights.append(w)
        logf_save.append(logf) 

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
        # 4.1 Sample covariance
        sample_Cov_0=[] 
        rob_Cov_1_0=[]   
        rob_Cov_2_0=[]
        all_weights_new = all_weights[t].copy()     
        for n in range(N):
            ind_nn = [i for i in range((n)*K, (n+1)*K)]
            samples_nn = all_samples[t][:,ind_nn]
            weights_nn = all_weights_new[ind_nn]
            norm_weights = weights_nn/np.sum(weights_nn)
            W_bessel_n = 1 - np.sum(norm_weights**2)
            sample_mean_nn = torch.sum(torch.tensor(norm_weights).cuda().reshape(1,len(norm_weights)).repeat(M, 1)*samples_nn,1)
            sample_Cov_nn = weightedcov(torch.transpose(samples_nn,0,1).double(),norm_weights) # is the sample covariance approximated # it's a symmetric matrix
            sample_Cov_0.append(sample_Cov_nn)
            norm_weights_cropped = crop_weights(norm_weights,fraction)
            sample_Cov_nn_cropped = weightedcov(torch.transpose(samples_nn,0,1).double(),norm_weights_cropped)#  sample_Cov_nn_cropped is (a possibly biased) stable estimator of the covariance matrix

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


        # 4.2 mean adaptation using Langevin
        ct = 0
        Save_A=[]
        samples_resampled_adapted = torch.zeros(M,N).cuda()
        Sigma_resampled = torch.zeros(M,M,N).cuda().double()
        gradient_save = []
        samples_save = []
        for n in range(N):
            samples_resampled_n = samples_resampled[:,n].reshape(M,1)
            [W_g,B_g,l] = evaluate_target_general(samples_resampled_n.double(),tp)
            if cov_type==0 :
                A_n = sample_Cov[t][n]
            elif cov_type==1:
                A_n=rob_Cov_1[t][n]
            elif cov_type==2 :    
                A_n=rob_Cov_2[t][n]    
            try:
                ff = multivariate_normal(np.zeros((M)),A_n.cpu().numpy()).pdf(np.zeros((M)))
                flag_A = 1
            except:
                flag_A = 0
            if flag_A == 1:
                if (np.isreal(sqrtm(A_n.cpu().numpy()))).all() == False:
                    flag_A = 0
            if flag_A == 0:
                ct = ct+1
                if t == 0:
                    Save_A.append(Sigma0_small.double().cuda())
                else:
                    parent_index_new = parents_samples_resampled[t][n]
                    Save_A.append(Past_A[parent_index_new])
            else:
                Save_A.append(A_n)  
            dirn = Param2vec(W_g,B_g,tp)
            dn =  Save_A[n]@dirn.double()
            mysteps = steps
            cts = 0
            f_0=l
            _,_,f_s = evaluate_target_general(samples_resampled_n-mysteps*dn,tp)
            while f_s > f_0 : #backtracking
                mysteps = mysteps*coef_step
                cts = cts + 1
                _,_,f_s = evaluate_target_general(samples_resampled_n-mysteps*dn,tp)  
            samples_resampled_adapted_0 = samples_resampled_n - mysteps*dn
            samples_resampled_adapted[:,n] = samples_resampled_adapted_0.squeeze(1)
            Sigma_resampled[:,:,n] = Save_A[n]
            gradient_save.append(dirn)
            samples_save.append(samples_resampled[:,n])
        Past_A=Save_A
        ct_vec.append(ct)
        print(['Iteration ',t,': we discard ',ct,'/',N,' Sigma matrices'])

    return Mean_est_evolution,Z_est_evolution,all_samples, all_weights,all_proposals,all_Sigma,mean_est_part,Z_part,m2_est_evolution,m2_part,map_estimator,logf_save,ct_vec,samples_resampled_adapted,Sigma_resampled