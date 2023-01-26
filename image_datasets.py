from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from sklearn.datasets import make_swiss_roll
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import collections
import json
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import imageio
from torchvision.utils import make_grid, save_image
import models_1d
import grammars
import train
import pickle


_learning_rate = 0.02
_momentum = 0
_alpha = 0.99
_weight_decay = 0.05
_eps = 1e-08
_dampening = 0
_weight_decay = 0
_batch_size = 512
_budget = 1000


torch.seed(800)

transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,)),
])


train_data = datasets.MNIST(
    root='../input/data',
    train=True,
    download=True,
    transform=transform
)
loader = DataLoader(train_data, batch_size=_batch_size, shuffle=True)




all_results = []
_trials = 1
for i in range(_trials):
    print(f"Trial {i + 1}/{_trials}")
    G = models_1d.Generator(latent_size=256, layer_size=784, num_layers=3, output_size=784, is_img=True)
    G.load_state_dict(torch.load(f"models/g_base_{i}.param"))
    #g_opt = optim.RMSprop(G.parameters(), _learning_rate, _alpha, _eps, _weight_decay, _momentum)
    #g_opt = optim.SGD(G.parameters(), _learning_rate, _momentum, _dampening, _weight_decay)
    g_opt  = optim.Adam(G.parameters(), lr=0.00001)

    D = models_1d.Discriminator(data_size=784, layer_size=500, num_layers=3, sig=False)
    D.load_state_dict(torch.load(f"models/d_base_{i}.param"))
    #d_opt = optim.RMSprop(D.parameters(), _learning_rate, _alpha, _eps, _weight_decay, _momentum)
    d_opt  = optim.Adam(D.parameters(), lr=0.00001)

    policy = grammars.ALT_GDA(3,1)
    # NEW_FLIPR_GDA(_X)
    # FLIPR_GDA(_X)
    # ALT_GDA(1,1)
    # ALT_GDA(5,1)
    # SIM_GDA()
    # RandomFlip()

    results = train.train(G, D, g_opt, d_opt, loader, _budget, policy, loss_type="WASSGP", clip=False, clip_range=(-1, 1))
    all_results.append(results)
    pickle.dump(all_results, open("mnist_ALT.p", "wb"))