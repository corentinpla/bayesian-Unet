import torch
from torch import nn
import numpy as np
import math
from torch.autograd import Variable
from scipy.stats import multivariate_normal
from scipy.linalg import sqrtm
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import logging
import torch.nn.functional as F
from scipy.stats import bernoulli, binom
import torch.optim as optim
from Model_files.LeNet5 import *


def OpenMat(x):
    """
    Converts a numpy array loaded from a .mat file into a properly ordered tensor.
    Parameters
    ----------
        x (numpy array): image loaded from a .mat file, size h*w*c, c in {2,3}   
    Returns
    -------
        (torch.FloatTensor): size c*h*w
    """
    return torch.from_numpy(x).type(torch.FloatTensor)


def to_variable(var=(), cuda=True, volatile=False):
    out = []
    for v in var:
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out



"""
Function: create a logger to save logs.
"""
def get_logger(log_file_name):
    logger=logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    handler=logging.FileHandler(log_file_name)
    handler.setLevel(logging.DEBUG)
    formatter=logging.Formatter('%(asctime)s: %(name)s (%(levelname)s)  %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


"""
Define a function to print and save log simultaneously.
"""
def myprint(str,logger):
    logger.info(str)
    print(str) 

def mvnpdf(x_0,mu,s):
    N = x_0.shape[0]
    fp_mix=[multivariate_normal(mu[i,:],s[:,:,i]).pdf(x_0[i,:]) for i in range(N)]
    #return fp_mix.clone().detach()
    return torch.tensor(fp_mix)

def is_pos_def(A): # Check if matrix A is defined as positive
    if np.array_equal(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def FFnetwork(W,b,L,X,activation):
    for ll in range(0,L):
        # dropout
        p = 1
        pdf = bernoulli(p)
        temp = torch.tensor((pdf.rvs(1))).cuda()
        W_temp = W[ll+1]*temp
        temp = torch.tensor((pdf.rvs(1))).cuda()
        b_temp = b[ll+1]*temp
        output = torch.mm(X.cuda(), W_temp) + b_temp.unsqueeze(0).expand(X.shape[0], -1)
        if activation[ll] == 'tanh':
            act = nn.Tanh()
            output = act(output)
        elif activation[ll] == 'sigmoid':
            act = nn.Sigmoid()
            output = act(output)
        elif activation[ll] == 'ReLU':
            act = nn.ReLU(inplace=True) 
            output = act(output)
        else:
            output = output
        X = output    
    return output


def Param2vec(W,b,tp):
    # combine W and b to a vector
    L = tp['L']
    M = tp['M']
    dimension = tp['dimension']
    vector = torch.zeros(dimension,1).cuda()
    ind = 0
    for ll in range(L):
        vector[ind:ind+M[ll]*M[ll+1],:]=torch.transpose(W[ll],0,1).reshape(M[ll]*M[ll+1],1)
        ind = M[ll]*M[ll+1]
        vector[ind:ind+M[ll+1],:]=b[ll].reshape(M[ll+1],1)
        ind = ind+M[ll+1]
    return vector


def weightedcov(Y, w):
    w = torch.tensor(w).cuda()
    w=w/w.sum()
    T, N = Y.size()   
    C = Y - (w @ Y).repeat(T, 1)   
    C = torch.transpose(C,0,1) @ (C* (w.unsqueeze(-1).repeat(1, N)))                                           
    C = 0.5 * (C + torch.transpose(C,0,1))
    return C

def diagweightedcov(Y, w):
    w = torch.tensor(w).cuda()
    w=w/w.sum()
    a = w @ Y  
    diagC = torch.sum(((Y-a)**2)*w.unsqueeze(-1),dim=0)
    return diagC

def crop_weights(norm_weights,fraction):
    norm_weights = norm_weights+1e-30
    K = len(norm_weights)
    d=norm_weights.tolist()
    d.sort(reverse=True)
    all_val=torch.tensor(d).cuda().double()
    #all_val = d.clone().detach()
    if fraction == 0 :
        ind = 0
    else:
        ind = math.ceil(fraction*K)
    max_val = all_val[ind]
    crop_norm_weights = torch.tensor(norm_weights).cuda().double()
    crop_norm_weights[crop_norm_weights>max_val] = max_val
    c = crop_norm_weights/torch.sum(crop_norm_weights)
    return c



def Vec2param(vector,tp):
    # save W and b as dictionary
    L = tp['L']
    M = tp['M']
    W = {}
    b = {}
    for ll in range(L):
        W[ll+1] = torch.transpose(vector[:M[ll]*M[ll+1]].reshape(M[ll+1],M[ll]),0,1)
        W[ll+1] = Variable(W[ll+1], requires_grad=True) 
        vector = vector[M[ll]*M[ll+1]:]
        b[ll+1] = vector[:M[ll+1]].reshape(M[ll+1])
        b[ll+1] = Variable(b[ll+1], requires_grad=True)
        vector = vector[M[ll+1]:]
    return W, b  

def functionR(Sigma,N):
    # only needs to be calculated for once
    d = Sigma.shape[0]
    logSqrtDetSigma = torch.zeros(N).cuda()
    inverse_sigma = torch.zeros(d,d,N).cuda()
    for i in range(N):
        R = torch.cholesky(Sigma[:,:,i], upper = True)
        inverse_sigma[:,:,i] = torch.inverse(R)
        logSqrtDetSigma[i] = torch.sum(torch.log(torch.diag(R)))
    return inverse_sigma, logSqrtDetSigma

def functionMu(x, mu, N):
    s = mu.shape[0]
    n = x.shape[0]
    x_new = torch.repeat_interleave(x, repeats=s, dim=0)
    mu_new = mu.repeat(n,1)
    temp = torch.split(x_new-mu_new,s)
    temp = torch.stack(list(temp))
    output = temp.permute(1, 2, 0)
    return output

def evaluate_proposal_multiple_fullCov(x, mu, Sig,N):
    d = x.shape[0]
    x=torch.transpose(x,0,1)
    mu=torch.transpose(mu,0,1)
    fp_mixt = []
    inverse_sigma, logSqrtDetSigma = functionR(Sig,N)
    logSqrtDetSigma1 = logSqrtDetSigma.double()
    sigma_constant =  torch.exp(-logSqrtDetSigma1 - d*np.log(2*np.pi)/2).cuda()
    X0 = functionMu(x, mu, N) #N*d*n
    xRinv = torch.einsum('lik,ijl->ljk',X0, inverse_sigma) # inverse_sigma: d*d*N
    quadform = (xRinv**2).sum(1)
    y = (torch.exp(-0.5*quadform).cuda()*(sigma_constant.unsqueeze(-1))).sum(0)
    fp_mixt = y/N
    return torch.tensor(fp_mixt)

def evaluate_regularization(weight, W, b, prior):
    if prior == 'L2':
        phi_W = 1/2*torch.sum(W**2)
        phi_b = 1/2*torch.sum(b**2)
        phi = phi_W + phi_b
        phi = phi*weight
    elif prior == 'L1':
        phi_W = 1/2*torch.sum(torch.abs(W))
        phi_b = 1/2*torch.sum(torch.abs(b))
        phi = phi_W + phi_b
        phi = phi*weight
    return phi


# the prediction of train data and test data for regression problem
def predict_regression(train_loader,net):
    z_est_train = [] # the predicted value
    y_train_new = []
    cost_train = []
    for x, y in train_loader:
        yhat, cost = net.eval(x, y)
        y_train_new.extend(y.tolist())
        z_est_train.extend(yhat.tolist())# yhat
        cost_train.extend(cost.tolist())# cost
    return z_est_train, y_train_new, cost_train



def criteria(is_binary,Y_train1,z_est_train,z_est_train1,y_train_new):
    # AUC calculation on train set
    if is_binary==0:
        auc_train,fpr_train,tpr_train,aucAv_train =Mul_AUC(Y_train1,torch.tensor(z_est_train1))
    elif is_binary==1:
        auc_train,fpr_train, tpr_train=fastAUC(Y_train1,z_est_train1)
        aucAv_train = 0
        
    # confusion matrix on train set
    z_est_train = (np.array(z_est_train)>0.5)*1.0
    matrix_train=confusion_matrix(np.array(y_train_new),z_est_train)

    TP = matrix_train[0,0]
    FN = matrix_train[0,1]
    FP = matrix_train[1,0]
    TN = matrix_train[1,1]

    Precision = TP/(TP+FP)
    Recall = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    Accuracy = (TP+TN)/(TP+TN+FP+FN)
    F1 = 2*TP/(2*TP+FP+FN)

    return auc_train,fpr_train,tpr_train,aucAv_train,matrix_train,Precision,Recall,Specificity,Accuracy,F1





def fastAUC(y,pred):
    fpr, tpr=my_roc_curve(y,pred)
    auc = metrics.auc(fpr, tpr)
    return auc, fpr, tpr

def my_roc_curve(labels, scores):
    n = labels.shape[0]
    num_pos = sum(labels)
    scores_si_reindex = np.argsort(-1*scores.reshape(n))
    l = labels[scores_si_reindex]
    tp = np.cumsum(l==1.0) # cumulative sum
    fp = np.array(range(1,n+1))-tp
    num_neg = n-num_pos
    fpr = fp/num_neg #False Positive Rate
    tpr = tp/num_pos #True Positive Rate
    return fpr, tpr



def BNN_posterior_regression(N, K, N_resampled,population, weights, x_test, y_test, tp):
    L =tp['L']
    weights = np.nan_to_num(weights)
    wn = weights/sum(weights)
    
    positions = np.random.choice([i for i in range(N*K)], N_resampled, replace = True, p = wn )

    z_particle_test_all = []
    resampled_particle = []
    MSE_test_particle_all = []
    for jj in range(N_resampled):
        #look at the obtained weights
        resampled_particle.append(population[:,positions[jj]])
        [W_particle,b_particle] = Vec2param(resampled_particle[jj],tp)

        z_particle_test = FFnetwork(W_particle,b_particle,L,x_test,tp['activation'])
        z_particle_test_all.append(z_particle_test) 
        
        loss_ = nn.MSELoss(reduction='mean')
        loss = loss_(z_particle_test,y_test.cuda()).detach().cpu().numpy()
        
        MSE_test_particle_all.append(loss)
            
    return  z_particle_test_all, resampled_particle, MSE_test_particle_all


def calculate_grad(Weight_0,B_0,tp):
    L = tp['L']
    X = tp['x_0'].cuda().double()
    y = tp['y'].cuda()
    prior_W = tp['prior_W']
    prior_b = tp['prior_b']
    regularization_weight = tp['regularization_weight']
    activation = tp['activation']
    classification = tp['classification']

    W1 = Variable(Weight_0[1], requires_grad=True)
    b1 = Variable(B_0[1], requires_grad=True)
    W2 = Variable(Weight_0[2], requires_grad=True)
    b2 = Variable(B_0[2], requires_grad=True)
    
    parameters = [W1, b1, W2, b2]
    
    optimizer = optim.Adam(parameters)
    
    W = {}
    b = {}
    
    W[1] = W1
    W[2] = W2
    b[1] = b1
    b[2] = b2
    
    output = FFnetwork(W,b,L,X,activation)
            
    if classification == 'binary':        
        y = y.float()
        loss_ = nn.BCEWithLogitsLoss(reduction='sum')
        loss = loss_(output, y)
    elif classification == 'regression':
        y = y.double()
        loss_ = nn.MSELoss(reduction='mean')
        loss = loss_(output, y)
    elif classification == 'regression_L1':
        y = y.double()  
        mae = nn.L1Loss()
        loss = mae(output, y)
    else:
        loss = F.cross_entropy(output, y, reduction='sum')# the MSE part
        
    l1_penalty = torch.tensor(0.0)  
    l2_penalty = torch.tensor(0.0)  
    if prior_W == 'L1':  
        l1_penalty = regularization_weight * sum([p.abs().sum() for p in parameters])
    elif prior_W == 'L2':    
        l2_penalty = regularization_weight * sum([(p**2).sum() for p in parameters])
    loss_with_penalty = loss + l1_penalty + l2_penalty
    
    
    optimizer.zero_grad()
    loss_with_penalty.backward()
    optimizer.step()
    
    W_g = {}
    B_g = {}
    
    W_g1 = optimizer.param_groups[0]['params'][0].grad
    W_g[0] = W_g1
    W_g2 = optimizer.param_groups[0]['params'][2].grad
    W_g[1] = W_g2
    B_g1 = optimizer.param_groups[0]['params'][1].grad
    B_g[0] = B_g1
    B_g2 = optimizer.param_groups[0]['params'][3].grad
    B_g[1] = B_g2

    
    return W_g,B_g,loss_with_penalty


def calculate_grad_diag(Weight_0,B_0,tp,x_train,y_train):
    L = tp['L']
    X = x_train.cuda().double()
    y = y_train.cuda()
    prior_W = tp['prior_W']
    prior_b = tp['prior_b']
    regularization_weight = tp['regularization_weight']
    activation = tp['activation']
    classification = tp['classification']

    W1 = Variable(Weight_0[1], requires_grad=True)
    b1 = Variable(B_0[1], requires_grad=True)
    W2 = Variable(Weight_0[2], requires_grad=True)
    b2 = Variable(B_0[2], requires_grad=True)
    
    parameters = [W1, b1, W2, b2]
    
    optimizer = optim.Adam(parameters)
    
    W = {}
    b = {}
    
    W[1] = W1
    W[2] = W2
    b[1] = b1
    b[2] = b2
    
    output = FFnetwork(W,b,L,X,activation)
            
    if classification == 'binary':        
        y = y.float()
        loss_ = nn.BCEWithLogitsLoss(reduction='sum')
        loss = loss_(output, y)
    elif classification == 'regression':
        y = y.double()
        loss_ = nn.MSELoss(reduction='mean')
        loss = loss_(output, y)
    elif classification == 'regression_L1':
        y = y.double()  
        mae = nn.L1Loss()
        loss = mae(output, y)
    else:
        loss = F.cross_entropy(output, y, reduction='sum')# the MSE part
        
    l1_penalty = torch.tensor(0.0)  
    l2_penalty = torch.tensor(0.0)  
    if prior_W == 'L1':  
        l1_penalty = regularization_weight * sum([p.abs().sum() for p in parameters])
    elif prior_W == 'L2':    
        l2_penalty = regularization_weight * sum([(p**2).sum() for p in parameters])
    loss_with_penalty = loss + l1_penalty + l2_penalty
    
    
    optimizer.zero_grad()
    loss_with_penalty.backward()
    optimizer.step()
    
    W_g = {}
    B_g = {}
    
    W_g1 = optimizer.param_groups[0]['params'][0].grad
    W_g[0] = W_g1
    W_g2 = optimizer.param_groups[0]['params'][2].grad
    W_g[1] = W_g2
    B_g1 = optimizer.param_groups[0]['params'][1].grad
    B_g[0] = B_g1
    B_g2 = optimizer.param_groups[0]['params'][3].grad
    B_g[1] = B_g2

    
    return W_g,B_g,loss_with_penalty


def evaluate_target_general(vector,tp):
    [Weight_0,B_0] = Vec2param(vector, tp)
    [W_g,B_g,l] = calculate_grad(Weight_0,B_0,tp)
    return W_g,B_g,l


def evaluate_target_general_diag(vector,tp,x_train,y_train):
    [Weight_0,B_0] = Vec2param(vector, tp)
    [W_g,B_g,l] = calculate_grad_diag(Weight_0,B_0,tp, x_train, y_train)
    return W_g,B_g,l

def calculate_grad_regression_large(Weight_0,B_0,tp):
    L = tp['L']
    X = tp['x_0'].cuda().double()
    y = tp['y'].cuda()
    prior_W = tp['prior_W']
    prior_b = tp['prior_b']
    regularization_weight = tp['regularization_weight']
    activation = tp['activation']
    classification = tp['classification']

    W1 = Variable(Weight_0[1], requires_grad=True)
    b1 = Variable(B_0[1], requires_grad=True)
    W2 = Variable(Weight_0[2], requires_grad=True)
    b2 = Variable(B_0[2], requires_grad=True)
    W3 = Variable(Weight_0[3], requires_grad=True)
    b3 = Variable(B_0[3], requires_grad=True)
    
    parameters = [W1, b1, W2, b2, W3, b3]
    
    optimizer = optim.Adam(parameters)
    
    W = {}
    b = {}
    
    W[1] = W1
    W[2] = W2
    W[3] = W3
    b[1] = b1
    b[2] = b2
    b[3] = b3
    
    output = FFnetwork(W,b,L,X,activation)
            
    if classification == 'binary':        
        y = y.float()
        loss_ = nn.BCEWithLogitsLoss(reduction='sum')
        loss = loss_(output, y)
    elif classification == 'regression':
        y = y.double()
        loss_ = nn.MSELoss(reduction='mean')
        loss = loss_(output, y)
    else:
        loss = F.cross_entropy(output, y, reduction='sum')# the MSE part
        
    l1_penalty = torch.tensor(0.0)  
    l2_penalty = torch.tensor(0.0)  
    if prior_W == 'L1':  
        l1_penalty = regularization_weight * sum([p.abs().sum() for p in parameters])
    elif prior_W == 'L2':    
        l2_penalty = regularization_weight * sum([(p**2).sum() for p in parameters])
    loss_with_penalty = loss + l1_penalty + l2_penalty
    
    
    optimizer.zero_grad()
    loss_with_penalty.backward()
    optimizer.step()
    
    W_g = {}
    B_g = {}
    
    W_g1 = optimizer.param_groups[0]['params'][0].grad
    W_g[0] = W_g1
    W_g2 = optimizer.param_groups[0]['params'][2].grad
    W_g[1] = W_g2
    W_g3 = optimizer.param_groups[0]['params'][4].grad
    W_g[2] = W_g3
    B_g1 = optimizer.param_groups[0]['params'][1].grad
    B_g[0] = B_g1
    B_g2 = optimizer.param_groups[0]['params'][3].grad
    B_g[1] = B_g2
    B_g3 = optimizer.param_groups[0]['params'][5].grad
    B_g[2] = B_g3
    
    return W_g,B_g,loss_with_penalty


def evaluate_target_general_regression_large(vector,tp):
    [Weight_0,B_0] = Vec2param(vector, tp)
    [W_g,B_g,l] = calculate_grad_regression_large(Weight_0,B_0,tp)
    return W_g,B_g,l


def Vec2param_LeNet5(vector,model):
    state_dict = model.state_dict()
    temp_vec = vector[:6*5*5].reshape(6,1,5,5)
    state_dict['conv1.weight'] = temp_vec
    vector = vector[6*5*5:]
    temp_vec = vector[:6].reshape(6)
    state_dict['conv1.bias'] = temp_vec
    vector = vector[6:]
    temp_vec = vector[:16*6*5*5].reshape(16,6,5,5)
    state_dict['conv2.weight'] = temp_vec
    vector = vector[16*6*5*5:]
    temp_vec = vector[:16].reshape(16)
    state_dict['conv2.bias'] = temp_vec
    vector = vector[16:]
    temp_vec = vector[:120*400].reshape(120,400)
    state_dict['fc1.weight'] = temp_vec
    vector = vector[120*400:]
    temp_vec = vector[:120].reshape(120)
    state_dict['fc1.bias'] = temp_vec
    vector = vector[120:]
    temp_vec = vector[:84*120].reshape(84,120)
    state_dict['fc2.weight'] = temp_vec
    vector = vector[84*120:]
    temp_vec = vector[:84].reshape(84)
    state_dict['fc2.bias'] = temp_vec
    vector = vector[84:]
    temp_vec = vector[:10*84].reshape(10,84)
    state_dict['fc3.weight'] = temp_vec
    vector = vector[10*84:]
    temp_vec = vector[:10].reshape(10)
    state_dict['fc3.bias'] = temp_vec
    vector = vector[10:]
    model.load_state_dict(state_dict)
    return model


def Param2vec_LeNet5(net,M):
    state_dict = net.model.state_dict()
    vector = torch.zeros(M)
    ind_total = 0

    ind_temp = 6*5*5
    temp = state_dict['conv1.weight'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 6
    temp = state_dict['conv1.bias'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 16*6*5*5
    temp = state_dict['conv2.weight'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 16
    temp = state_dict['conv2.bias'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 120*400
    temp = state_dict['fc1.weight'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 120
    temp = state_dict['fc1.bias'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 84*120
    temp = state_dict['fc2.weight'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 84
    temp = state_dict['fc2.bias'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 10*84
    temp = state_dict['fc3.weight'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp

    ind_temp = 10
    temp = state_dict['fc3.bias'].reshape(ind_temp)
    vector[ind_total:ind_total+ind_temp] = temp
    ind_total = ind_total+ind_temp
    return vector


        

def diagevaluate_proposal_multiple_fullCov(x, mu, diagSig,N):
    d = x.shape[0]
    n = x.shape[1]
    K = n//N
    x=torch.transpose(x,0,1)
    mu=torch.transpose(mu,0,1)
    fp_mixt = []
    inverse_sigma, logSqrtDetSigma = diagfunctionR(diagSig,N)
    logSqrtDetSigma1 = logSqrtDetSigma.double()
    fp_mixt = torch.tensor(()).cuda()
    for k in range(K):
        x_temp = x[k*N:(k+1)*N,:]
        X0 = functionMu(x_temp, mu, N) #N*d*n
        xRinv = torch.einsum('lik,il->lik',X0, inverse_sigma.cuda()) # inverse_sigma: d*d*N
        quadform = (xRinv**2).sum(1)
        y_temp = -0.5*quadform-logSqrtDetSigma1.cuda()
        if k == 0:
            C = -y_temp.max()
        y_temp = y_temp + C
        y = (torch.exp(y_temp)).sum(0)
        fp_mixt = torch.cat((fp_mixt,y),0)
    return fp_mixt


def diagfunctionR(diagSigma,N):
    # only needs to be calculated for once
    inverse_sigma = diagSigma**(-1/2)
    logSqrtDetSigma = torch.sum(torch.log(diagSigma**(1/2)),dim=0)
    return inverse_sigma, logSqrtDetSigma

