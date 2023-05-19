import torch.nn as nn
import torch
import math
from einops import rearrange

class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, init_filters=32, depth=3, time_embedding_dim=32, device="cuda:0"):
        super(UNet1D, self).__init__()

        self.time_embedding_dim = time_embedding_dim
        self.depth = depth
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.device = device

        pos_embedding = SinusoidalPosEmb(self.time_embedding_dim)

        self.time_mlp = nn.Sequential(
            pos_embedding,
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.GELU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim)
        )
        
        # Initialize number of filters
        filters = init_filters
        
        # Define downsampling blocks
        for _ in range(self.depth):
            self.downs.append(DownsampleBlock(in_channels, filters, time_embedding_dim=self.time_embedding_dim))
            in_channels = filters
            filters *= 2
            
        # Define bottom block
        self.bottom = nn.Sequential(
            nn.Conv1d(filters//2, filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True),
            Attention(dim=filters, dim_head=64, heads=12),
            nn.Conv1d(filters, filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(filters),
            nn.ReLU(inplace=True)
        )
        
        # Define upsampling blocks
        for _ in reversed(range(self.depth)):
            self.ups.append(UpsampleBlock(filters, in_channels, time_embedding_dim=self.time_embedding_dim))
            filters //= 2
            in_channels = filters // 2
            
        # Define output layer
        self.output = nn.Conv1d(init_filters, out_channels, kernel_size=1)
        
    def forward(self, x, t):
        intermediates = []
        t = self.time_mlp(t.to(self.device))
        
        # Downsampling path
        for down in self.downs:
            x, inter = down(x, t)
            intermediates.append(inter)
            
        x = self.bottom(x)
        
        # Upsampling path
        for i, up in enumerate(self.ups):
            inter = intermediates[-(i+1)]
            x = up(x, inter, t)

        out = self.output(x)
        
        return out

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim=None):
        super(DownsampleBlock, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, in_channels)
        )

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x, time_embedding):
        time_embedding = self.mlp(time_embedding)
        time_embedding = rearrange(time_embedding, 'b c -> b c 1')

        x = x + time_embedding
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        out = self.pool(x)

        return out, x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim=None):
        super(UpsampleBlock, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embedding_dim, out_channels)
        )

        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x, inter, time_embedding):
        time_embedding = self.mlp(time_embedding)
        time_embedding = rearrange(time_embedding, 'b c -> b c 1')

        x = self.upsample(x)
        x = x + time_embedding
        x = torch.cat([x, inter], dim=1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x
    
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv1d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale        

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c n -> b (h c) n', h = self.heads)

        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h = self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b (h d) n')

        return self.to_out(out)
    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)

        return (x - mean) * (var + eps).rsqrt() * self.g