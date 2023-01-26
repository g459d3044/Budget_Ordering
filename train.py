from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
import models
import math


def get_stats(G, D, given_data):
    with torch.no_grad():
        latent_noise = len(given_data)
        fake_sample = G(latent_noise)
        fake_projections = D(fake_sample)
        real_projections = D(given_data)

        mean_fake = torch.mean(fake_projections)
        mean_real = torch.mean(real_projections)

        real_projs_flt = torch.flatten(fake_projections)
        fake_projs_flt = torch.flatten(real_projections)

        # Wasserstein Distances
        w_distD = wasserstein_distance(real_projs_flt, fake_projs_flt)


        # Support Alignment
        fake_supp = (torch.min(fake_projs_flt), torch.max(fake_projs_flt))
        real_supp = (torch.min(real_projs_flt), torch.max(real_projs_flt))
        overlap_supp = (-1, -1)
        if (fake_supp[1] > real_supp[0]) and (real_supp[1] > fake_supp[0]):
            overlap_supp = (torch.max(fake_supp[0], real_supp[0]).item(), torch.min(fake_supp[1], real_supp[1]).item())
        support_union = (real_supp[1] - real_supp[0]) + (fake_supp[1] - fake_supp[0])
        support_alignment = (overlap_supp[1] - overlap_supp[0]) / support_union

        # Mass in overlaps
        real_mass_in_fake = 0
        fake_mass_in_real = 0
        if overlap_supp[0] != -1:
            fake_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(fake_projs_flt.reshape(-1,1))
            real_mass_in_fake = math.exp(fake_kde.score(real_projs_flt.reshape(-1,1)))

            real_kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(real_projs_flt.reshape(-1,1))
            fake_mass_in_real = math.exp(real_kde.score(fake_projs_flt.reshape(-1,1)))
    return w_distD, real_supp, fake_supp, overlap_supp, support_alignment, real_mass_in_fake, fake_mass_in_real, mean_real, mean_fake


class Sampler:
    def __init__(self, loader):
        self.loader = loader
        self.iterator = iter(self.loader)
        self.blocks = len(self.loader)
        self.current = 0
    def get_sample(self):
        if self.current == self.blocks:
            self.current = 0
            self.iterator = iter(self.loader)
        train_features = next(self.iterator)[0]
        self.current += 1
        return train_features


def train(G, D, G_opt, D_opt, real_data, budget, GDA_policy, loss_type, clip = False, clip_range = None, gp_penalty = 10, record_synthetic_samples=25):
    # statistics to keep as a list for the number of epochs
    wass_dists_D = []
    # what the actual GDA-policy chose
    policy_choice = []
    # loss
    gen_losses = []
    disc_losses = []
    # properties of distributions in D's output space
    real_support = []
    synthetic_support = []
    overlapping_support = []
    support_alignment = []
    fake_mass_in_overlap = []
    real_mass_in_overlap = []
    mean_real_proj = []
    mean_fake_proj = []
    synthetic_samples = []

    data_sampler = Sampler(real_data)

    # Pre_training record keeping
    w_distD, real_supp, fake_supp, overlap_supp, supp_alignment, real_mass_in_fake, fake_mass_in_real, mean_real, mean_fake = get_stats(G, D, data_sampler.get_sample())

    wass_dists_D.append(w_distD)
    real_support.append(real_supp)
    synthetic_support.append(fake_supp)
    overlapping_support.append(overlap_supp)
    support_alignment.append(supp_alignment)
    fake_mass_in_overlap.append(fake_mass_in_real)
    real_mass_in_overlap.append(real_mass_in_fake)
    mean_real_proj.append(mean_real)
    mean_fake_proj.append(mean_fake)
    G.eval()
    synthetic_samples.append(G(record_synthetic_samples))
    G.train()
    # start training looop
    pbar = tqdm(total=budget)
    current_iteration = 0
    while current_iteration < budget:
        real_sample = data_sampler.get_sample()
        machine_choice = GDA_policy.decide(G, D, current_iteration, budget, real_sample)


        g_loss = 0
        d_loss = 0
        G_opt.zero_grad()
        D_opt.zero_grad()
        # Train Generator
        if (machine_choice == 0) or (machine_choice == 2):
            g_loss = 0
            if loss_type == "WASS" or loss_type == "WASSGP":
                g_loss = models.get_gen_loss_Wasserstein(G, D, len(real_sample))
            elif loss_type == "BCE":
                g_loss = models.get_gen_loss_BCE(G, D, len(real_sample))
            gen_losses.append(g_loss)
            g_loss.backward()
        # Train Discriminator
        if (machine_choice == 1) or (machine_choice == 2):
            d_loss = 0
            if loss_type == "WASS":
                d_loss = models.get_disc_loss_Wasserstein(G, D, real_sample, len(real_sample), penalty=False)
            elif loss_type == "WASSGP":
                d_loss = models.get_disc_loss_Wasserstein(G, D, real_sample, len(real_sample), penalty=True)
            elif loss_type == "BCE":
                d_loss = models.get_disc_loss_BCE(G, D, real_sample, len(real_sample))
            disc_losses.append(d_loss)
            d_loss.backward()

        # Backprop Step
        if (machine_choice == 0) or (machine_choice == 2):
            G_opt.step()
        if (machine_choice == 1) or (machine_choice == 2):
            D_opt.step()
            if clip:
                for p in D.parameters():
                    p.data.clamp_(clip_range[0], clip_range[1])

        # Record Keeping
        w_distD, real_supp, fake_supp, overlap_supp, supp_alignment, real_mass_in_fake, fake_mass_in_real, mean_real, mean_fake = get_stats(G, D, data_sampler.get_sample())

        wass_dists_D.append(w_distD)
        real_support.append(real_supp)
        synthetic_support.append(fake_supp)
        overlapping_support.append(overlap_supp)
        support_alignment.append(supp_alignment)
        fake_mass_in_overlap.append(fake_mass_in_real)
        real_mass_in_overlap.append(real_mass_in_fake)
        mean_real_proj.append(mean_real)
        mean_fake_proj.append(mean_fake)
        synthetic_samples.append(G(record_synthetic_samples).detach())
        policy_choice.append(machine_choice)

        # Iterate training loop
        current_iteration += 1
        pbar.update(1)
        if machine_choice == 2:
            wass_dists_D.append(w_distD)
            real_support.append(real_supp)
            synthetic_support.append(fake_supp)
            overlapping_support.append(overlap_supp)
            support_alignment.append(supp_alignment)
            fake_mass_in_overlap.append(fake_mass_in_real)
            real_mass_in_overlap.append(real_mass_in_fake)
            mean_real_proj.append(mean_real)
            mean_fake_proj.append(mean_fake)
            synthetic_samples.append(G(record_synthetic_samples).detach())
            policy_choice.append(machine_choice)
            current_iteration += 1
            pbar.update(1)

    # Package stats as a dictionary and return
    pbar.close()
    train_session_dict = {}
    train_session_dict["wasserstein_distance_D"] = wass_dists_D
    train_session_dict["policy_choice"] = policy_choice
    train_session_dict["Generator_loss"] = gen_losses
    train_session_dict["Discriminator_loss"] = disc_losses
    train_session_dict["real_support_interval"] = real_support
    train_session_dict["synthetic_support_interval"] = synthetic_support
    train_session_dict["overlapping_support_interval"] = overlapping_support
    train_session_dict["support_alignment"] = support_alignment
    train_session_dict["real_mass_in_overlap"] = real_mass_in_overlap
    train_session_dict["synthetic_mass_in_overlap"] = fake_mass_in_overlap
    train_session_dict["synthetic_samples"] = synthetic_samples

    return train_session_dict