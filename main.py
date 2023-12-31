import math
from inspect import isfunction
from functools import partial
import random
import IPython

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk

from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from torch.optim import Adam

from torchvision.transforms import Compose, ToTensor, Lambda, ToPILImage, CenterCrop, Resize
import matplotlib.pyplot as plt

from model import Unet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cuda2 = torch.device('cuda:0')


# Reproductibility
torch.manual_seed(53)
random.seed(53)
np.random.seed(53)

# load hugging face dataset from the DSDIR
def get_dataset(data_path, batch_size, test = False):
    
    dataset = load_dataset(data_path)
    
    # define image transformations (e.g. using torchvision)
    transform = Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),  # Transform PIL image into tensor of value between [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1)  # Normalize values between [-1,1]
    ])

   # define function for HF dataset transform
    def transforms_im(examples):
        examples['pixel_values'] = [transform(image) for image in examples['image']]
        del examples['image']
        return examples

    dataset = dataset.with_transform(transforms_im).remove_columns('label')  # We don't need it 
    channels, image_size, _ = dataset['train'][0]['pixel_values'].shape
        
    if test:
        dataloader = DataLoader(dataset['test'], batch_size=batch_size)
    else:
        dataloader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)

    len_dataloader = len(dataloader)
    print(f"channels: {channels}, image dimension: {image_size}, len_dataloader: {len_dataloader}")  
    
    return dataloader, channels, image_size, len_dataloader


#Allow to see batch of images 

def normalize_im(images):
    shape = images.shape
    images = images.view(shape[0], -1)
    images -= images.min(1, keepdim=True)[0]
    images /= images.max(1, keepdim=True)[0]
    return images.view(shape)

def show_images(batch,ref):
    plt.imshow(torch.permute(make_grid(normalize_im(batch)), (1,2,0)))
    plt.savefig("/workspace/code/results/Diffusion/"+str(ref)+".png")


# Different type of beta schedule
def linear_beta_schedule(timesteps, beta_start = 0.0001, beta_end = 0.02):
    """
    linar schedule from the original DDPM paper https://arxiv.org/abs/2006.11239
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

# Function to get alphas and betas
def get_alph_bet(timesteps, schedule=cosine_beta_schedule):
    
    # define beta
    betas = schedule(timesteps)

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0) # cumulative product of alpha
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # corresponding to the prev const
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    const_dict = {
        'betas': betas,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'posterior_variance': posterior_variance
    }
    
    return const_dict

# extract the values needed for time t
def extract(constants, batch_t, x_shape):
    diffusion_batch_size = batch_t.shape[0]
    
    # get a list of the appropriate constants of each timesteps
    out = constants.gather(-1, batch_t.cpu()) 
    
    return out.reshape(diffusion_batch_size, *((1,) * (len(x_shape) - 1))).to(batch_t.device)


# forward diffusion (using the nice property)
def q_sample(constants_dict, batch_x0, batch_t, noise=None):
    if noise is None:
        noise = torch.randn_like(batch_x0)

    sqrt_alphas_cumprod_t = extract(constants_dict['sqrt_alphas_cumprod'], batch_t, batch_x0.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        constants_dict['sqrt_one_minus_alphas_cumprod'], batch_t, batch_x0.shape
    )

    return sqrt_alphas_cumprod_t * batch_x0 + sqrt_one_minus_alphas_cumprod_t * noise



@torch.no_grad() 
def p_sample(constants_dict, batch_xt, predicted_noise, batch_t):
    # We first get every constants needed and send them in right device
    betas_t = extract(constants_dict['betas'], batch_t, batch_xt.shape).to(batch_xt.device)
    sqrt_one_minus_alphas_cumprod_t = extract(
        constants_dict['sqrt_one_minus_alphas_cumprod'], batch_t, batch_xt.shape
    ).to(batch_xt.device)
    sqrt_recip_alphas_t = extract(
        constants_dict['sqrt_recip_alphas'], batch_t, batch_xt.shape
    ).to(batch_xt.device)
    
    # Equation 11 in the ddpm paper
    # Use predicted noise to predict the mean (mu theta)
    model_mean = sqrt_recip_alphas_t * (
        batch_xt - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )
    
    # We have to be careful to not add noise if we want to predict the final image
    predicted_image = torch.zeros(batch_xt.shape).to(batch_xt.device)
    t_zero_index = (batch_t == torch.zeros(batch_t.shape).to(batch_xt.device))
    
    # Algorithm 2 line 4, we add noise when timestep is not 1:
    posterior_variance_t = extract(constants_dict['posterior_variance'], batch_t, batch_xt.shape)
    noise = torch.randn_like(batch_xt)  # create noise, same shape as batch_x
    predicted_image[~t_zero_index] = model_mean[~t_zero_index] + (
        torch.sqrt(posterior_variance_t[~t_zero_index]) * noise[~t_zero_index]
    ) 
    
    # If t=1 we don't add noise to mu
    predicted_image[t_zero_index] = model_mean[t_zero_index]
    
    return predicted_image

# Algorithm 2 (including returning all images)
@torch.no_grad()
def sampling(model, shape, T, constants_dict):
    b = shape[0]
    # start from pure noise (for each example in the batch)
    batch_xt = torch.randn(shape, device=DEVICE)
    
    batch_t = torch.ones(shape[0]) * T  # create a vector with batch-size time the timestep
    batch_t = batch_t.type(torch.int64).to(DEVICE)
    
    imgs = []

    for t in tqdm(reversed(range(0, T)), desc='sampling loop time step', total=T):
        batch_t -= 1
        predicted_noise = model(batch_xt, batch_t)
        
        batch_xt = p_sample(constants_dict, batch_xt, predicted_noise, batch_t)
        
        imgs.append(batch_xt.cpu())
        
    return imgs

# Dataset parameters
batch_size = 64
data_path = "fashion_mnist"  
train_dataloader, channels, image_size, len_dataloader = get_dataset(data_path, batch_size)



epochs = 3
T = 1000  # = T
constants_dict = get_alph_bet(T, schedule=linear_beta_schedule)

model = Unet(   
    dim=image_size,
    init_dim=None,
    out_dim=None,
    dim_mults=(1, 2, 4),
    channels=channels,
    with_time_emb=True,
    convnext_mult=2,
).to(DEVICE)

criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4) #lr : learning rate 

####################################classic version##########################################
for epoch in range(epochs):
    loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
    for batch in loop:
        optimizer.zero_grad()

        batch_size_iter = batch["pixel_values"].shape[0]
        batch_image = batch["pixel_values"].to(DEVICE)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        batch_t = torch.randint(0, T, (batch_size_iter,), device=DEVICE).long()
        
        noise = torch.randn_like(batch_image)
        
        x_noisy = q_sample(constants_dict, batch_image, batch_t, noise=noise)
        predicted_noise = model(x_noisy, batch_t)
        
        loss = criterion(noise, predicted_noise)

        loop.set_postfix(loss=loss.item())

        loss.backward()
        optimizer.step() #mettre à jour les poids du model 

        sd = model.state_dict()
        # for param_tensor in model.state_dict():

        #     weights = model.state_dict()[param_tensor]
        #     print("before",weights)
        #     # Replace this part with the sampling ... 
        #     new_weigths = torch.zeros_like(weights)

        #     # reload the state_dict 
        #     sd[param_tensor] = new_weigths
        #     model.load_state_dict(sd)
        #     print("after", model.state_dict()[param_tensor])




print("check generation:")  
list_gen_imgs = sampling(model, (batch_size, channels, image_size, image_size), T, constants_dict)
show_images(list_gen_imgs[-1],1)

#make gif 
def make_gif(frame_list):
    to_pil = ToPILImage()
    frames = [to_pil(make_grid(normalize_im(tens_im))) for tens_im in frame_list]
    frame_one = frames[0]
    frame_one.save("sampling.gif.png", format="GIF", append_images=frames[::5], save_all=True, duration=10, loop=0)
    
    return IPython.display.Image(filename="./sampling.gif.png")