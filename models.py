import torch
import torch.nn as nn
from torch import autograd
import numpy as np


# Latent distribution
def get_noise(sample_size, latent_size):
    return torch.randn((sample_size, latent_size))


class Generator(nn.Module):
    def __init__(self, latent_size, layer_size, num_layers, output_size, is_img = False):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        if num_layers == 1:
            self.layers = [nn.Linear(latent_size, output_size, bias=True)]
        else:
            self.layers = [nn.Linear(latent_size, layer_size, bias=True)]
            for i in range(num_layers - 2):
                self.layers.append(nn.LeakyReLU(0.2))
                self.layers.append(nn.Linear(layer_size, layer_size, bias=True))
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.Linear(layer_size, output_size, bias=True))
        if is_img:
            self.layers.append(nn.Tanh())
        self.gen = nn.Sequential(*(self.layers))

    def forward(self, sample_size):
        return self.gen(get_noise(sample_size, self.latent_size))


def get_gen_loss_Wasserstein(gen, disc, synthetic_sample_size):
    fake_sample = gen(synthetic_sample_size)
    preds = disc(fake_sample)
    gen_loss = -torch.mean(preds)
    return gen_loss


def get_gen_loss_BCE(gen, disc, synthetic_sample_size):
    fake_sample = gen(synthetic_sample_size)
    preds = disc(fake_sample)
    gen_loss = nn.BCELoss()(preds, torch.ones_like(preds))
    return gen_loss


class Discriminator(nn.Module):
    def __init__(self, data_size, layer_size, num_layers, sig, is_img=True):
        super(Discriminator, self).__init__()
        self.sig = sig
        self.data_size = data_size
        self.is_img = is_img
        if num_layers == 1:
            self.layers = [nn.Linear(data_size, 1, bias=True)]
        else:
            self.layers = [nn.Linear(data_size, layer_size, bias=True)]
            for i in range(num_layers - 2):
                self.layers.append(nn.LeakyReLU(0.2))
                self.layers.append(nn.Linear(layer_size, layer_size, bias=True))
            self.layers.append(nn.LeakyReLU(0.2))
            self.layers.append(nn.Linear(layer_size, 1, bias=True))
        self.disc = nn.Sequential(*(self.layers))

    def forward(self, x):
        if self.is_img:
            x = x.view(-1, self.data_size)
        if self.sig:
            return nn.Sigmoid()(self.disc(x))
        else:
            return self.disc(x)


def compute_gp(netD, real_data, fake_data):
    batch_size = real_data.size(0)
    # Sample Epsilon from uniform distribution
    if real_data.dim() > 3:
        real_dat = real_data.view(-1, netD.data_size)
    else:
        real_dat = real_data
    eps = torch.rand(batch_size, 1).to(real_dat.device).requires_grad_()
    eps = eps.expand_as(real_dat)

    # Interpolation between real data and fake data.
    interpolation = eps * real_dat + (1 - eps) * fake_data

    # get logits for interpolated images
    interp_logits = netD(interpolation).requires_grad_()
    grad_outputs = torch.ones_like(interp_logits)

    # Compute Gradients
    gradients = autograd.grad(
        outputs=interp_logits,
        inputs=interpolation,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    # Compute and return Gradient Norm
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, 1)
    return torch.mean((grad_norm - 1) ** 2)


def get_disc_loss_Wasserstein(gen, disc, real_batch, synthetic_sample_size, penalty, gp_penalty=10):
    fake_sample = gen(synthetic_sample_size).detach()
    fake_preds = disc(fake_sample)
    loss_fake = torch.mean(fake_preds)
    real_preds = disc(real_batch)
    loss_real = torch.mean(real_preds)
    disc_loss = loss_fake - loss_real
    if penalty:
        disc_loss += gp_penalty*compute_gp(disc, real_batch, fake_sample)
    return disc_loss


def get_disc_loss_BCE(gen, disc, real_batch, synthetic_sample_size):
    fake_sample = gen(synthetic_sample_size).detach()
    fake_preds = disc(fake_sample)
    loss_fake = nn.BCELoss()(fake_preds, torch.zeros_like(fake_preds))
    real_preds = disc(real_batch)
    loss_real = nn.BCELoss()(real_preds, torch.ones_like(real_preds))
    disc_loss = loss_fake + loss_real
    return disc_loss