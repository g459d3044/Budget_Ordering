import torch
import models

_trials = 10
for i in range(_trials):
    G_base = models.Generator(latent_size=2, layer_size=16, num_layers=4, output_size=2)
    D_base = models.Discriminator(data_size=2, layer_size=16, num_layers=4, sig=False)
    #G_base = models_1d.Generator(latent_size=256, layer_size=784, num_layers=3, output_size=784, is_img=True)
    #D_base = models_1d.Discriminator(data_size=784, layer_size=500, num_layers=3, sig=False)
    torch.save(G_base.state_dict(), f"models_2d/g_base_{i}.param")
    torch.save(D_base.state_dict(), f"models_2d/d_base_{i}.param")