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
import models
import grammars
import train
import pickle

torch.manual_seed(800)



policies = [grammars.ALT_GDA(1,1), grammars.ALT_GDA(5,1), grammars.SIM_GDA(), grammars.RandomFlip(0.5), grammars.RandomFlip(0.8), grammars.FLIPR_GDA()]
names = ["alt11", "alt51", "sim", "random50", "random80", "flipr"]
#_learning_rate = [0.05, 0.005, 0.0001, 0.00005]
_learning_rate = [0.001]
_momentum = 0
_alpha = 0.99
_eps = 1e-08
_dampening = 0
_weight_decay = 0
_batch_size = 512
_budget = 25000


train_features = torch.tensor(pickle.load(open("datasets/ring_dataset.p", "rb")).clone().detach())
dataset = torch.utils.data.TensorDataset(train_features)
loader = torch.utils.data.DataLoader(dataset, batch_size=_batch_size, shuffle=True, drop_last=True)

for k in range(len(policies)):
    for j in range(len(_learning_rate)):
        _trials = 10
        for i in range(_trials):
            print(f"Trial {i + 1}/{_trials}")
            G = models.Generator(latent_size=2, layer_size=16, num_layers=4, output_size=2)
            G.load_state_dict(torch.load(f"models_2d/g_base_{i}.param"))
            g_opt = optim.Adam(G.parameters(), _learning_rate[j])

            D = models.Discriminator(data_size=2, layer_size=16, num_layers=4, sig=False)
            D.load_state_dict(torch.load(f"models_2d/d_base_{i}.param"))
            d_opt = optim.Adam(D.parameters(), _learning_rate[j])

            policy = policies[k]
            # NEW_FLIPR_GDA(_X)
            # FLIPR_GDA(_X)
            # ALT_GDA(1,1)
            # ALT_GDA(5,1)
            # SIM_GDA()
            # RandomFlip()

            results = train.train(G, D, g_opt, d_opt, loader, _budget, policy, loss_type="WASS", clip=True, clip_range=(-0.01, .01), record_synthetic_samples = 250)
            pickle.dump(results, open(f"experiment_data/gauss_8ring_{names[k]}_WASS_{i}.p", "wb"))