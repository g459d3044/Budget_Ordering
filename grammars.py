import torch
import train


# 0 : Train Generator
# 1 : Train Discriminator
# 2 : Train Both

# Simultaneous Gradient Descent-Ascent
# Always yields
class SIM_GDA():
    def __init__(self):
        pass
    def decide(self, gen, disc, i, B, real_batch):
        return 2


# Alternating Gradient Descent-Ascent
# Yields the follwing repeating pattern: ([1]^alpha * [0]^beta)
class ALT_GDA():
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.cycle = alpha + beta
    def decide(self, gen, disc, i, B, real_batch):
        tick = i % self.cycle
        decision = ""
        if tick < self.alpha:
            decision = 1
        else:
            decision = 0
        return decision


# Random Flip
# Yields a random sequence of 0's and 1's according to a weighted bernoulli distribution
class RandomFlip():
    def __init__(self, weighting):
        self.weighting = weighting
    def decide(self, gen, disc, i, B, real_batch):
        return torch.bernoulli(torch.tensor(self.weighting)).item()


# Random Walk
# Yields a random sequence of 0's, 1's, and 2's according to a uniform distribution
class RandomWalk():
    def __init__(self):
        pass
    def decide(self, gen, disc, i, B, real_batch):
        decision = ""
        rho = torch.rand(1)
        if rho < (1 / 3):
            decision = 0
        elif rho < (2 / 3):
            decision = 1
        else:
            decision = 2
        return decision


# You-Only-Train-Once
# Yields a sequnce of (00...0)^(B/2)(11...1)^(B/2) OR (11...1)^(B/2)(00...0)^(B/2)
class YOTO():
    def __init__(self, first, second):
        self.first = first
        self.second = second
    def decide(self, gen, disc, i, B, real_batch):
        decision = ""
        if (i / B) < 0.5:
            decision = self.first
        else:
            decision = self.second
        return decision


# FLIPR Gradient Descent-Ascent
class FLIPR_GDA():
    def __init__(self):
        pass
    def decide(self, gen, disc, i, B, real_batch):
        bern_lambda = self.continuous_decide(gen, disc, i, B, real_batch)
        return torch.bernoulli(torch.tensor(bern_lambda)).item()
    def continuous_decide(self, gen, disc, i, B, real_batch):
        w_distD, real_supp, fake_supp, overlap_supp, supp_alignment, real_mass_in_fake, fake_mass_in_real, mean_real, mean_fake = train.get_stats(gen, disc, real_batch)
        bern_lambda = fake_mass_in_real + supp_alignment - (2 * fake_mass_in_real * supp_alignment)
        return bern_lambda.item()


# FLIPR Gradient Descent-Ascent Deterministic
# Yields the rounded FLIPR_GDA
class FLIPR_GDA_D():
    def __init__(self):
        self.FLIPR = FLIPR_GDA()
    def decide(self, gen, disc, i, B, real_batch):
        return round(self.FLIPR.continuous_decide(gen, disc, i, B, real_batch))
