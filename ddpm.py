import torch
import torch.nn as nn
from tqdm import tqdm

class DDPM(nn.Module):
    def __init__(self, network, timesteps=1000, min_beta=10 ** -4, max_beta=0.02, input_shape=(256, 301)):
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        self.device = network.device
        self.image_chw = input_shape
        self.network = network
        self.betas = self.cosine_beta_schedule().to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(self.device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        x0 = x0.to(self.device)
        t = t.to(self.device)
        n, c, l = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, l).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1) * eta

        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

    @torch.no_grad()
    def sample(self, batch_size=1, channels=256, seq_length=301):
        x = torch.randn(batch_size, channels, seq_length).to(self.device)
        samples = []

        for i in tqdm(reversed(range(self.timesteps)), desc="Sampling timestep: ", leave=True, total=self.timesteps):
            t = torch.full((batch_size,), i, device=self.device, dtype=torch.long).to(self.device)
            pred = self.backward(x, t)

            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_bar_t).sqrt() * pred)

            if t > 0:
                z = torch.randn(batch_size, channels, seq_length).to(self.device)
                beta_t = self.betas[t]
                sigma_t = beta_t.sqrt()

                x = x + sigma_t * z

            samples.append(x)

        return samples
        
    def cosine_beta_schedule(self, s=0.008):
            steps = self.timesteps + 1
            x = torch.linspace(0, self.timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

            return torch.clip(betas, 0.0001, 0.9999)

