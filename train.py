import torch
import argparse
import wandb
import os
from datasets import FMA
from unet_1d import UNet1D
from ddpm import DDPM
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog="1D DDPM Train Script"
)

parser.add_argument('--wandb', default=True, required=False)
parser.add_argument('--epochs', default=2000, required=False)
parser.add_argument('--epochs_pred', default=5, required=False)
parser.add_argument('--timesteps', default=4000, required=False)
parser.add_argument('--initial_dim', default=128, required=False)
parser.add_argument('--depth', default=4, required=False)
parser.add_argument('--batch_size', default=32, required=False)
parser.add_argument('--n_fft', default=4096, required=False)
parser.add_argument('--n_mels', default=256, required=False)
parser.add_argument('--hop_length', default=2585, required=False)
parser.add_argument('--time_embedding_dim', default=256, required=False)
parser.add_argument('--seq_length', default=512, required=False)
parser.add_argument('--verbose', default=True, required=False)
parser.add_argument('--optimizer', default="AdamW", required=False)
parser.add_argument('--l_r', default=1e-4, required=False)
parser.add_argument('--eps', default=1e-6, required=False)
parser.add_argument('--weight_decay', default=1e-3, required=False)

def main():
    device = "cuda:0"
    args = parser.parse_args()
    dataset = FMA(workers=cpu_count(), sample_shape=(args.n_mels, args.seq_length))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    loss_fn = torch.nn.MSELoss()
    model = UNet1D(
        in_channels=args.n_mels,
        out_channels=args.n_mels,
        depth=args.depth,
        time_embedding_dim=args.time_embedding_dim,
        init_filters=args.initial_dim
    ).to(device)
    ddpm = DDPM(
        model,
        timesteps=args.timesteps,
        input_shape=(args.n_mels, args.seq_length)
    )
    if args.verbose:
        print(f"INFO: Number of trainable parameters: {sum(p.numel() for p in ddpm.parameters() if p.requires_grad)}")
    if args.verbose:
        print(f"INFO: Length of dataset: {len(dataset)}")

    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(ddpm.parameters(), lr=args.l_r, eps=args.eps, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.l_r, eps=args.eps, weight_decay=args.weight_decay)

    if args.wandb:
        wandb.login()
        wandb.init(
            project="confident-diffusion",
            config={
                "Epochs": args.epochs,
                "Timesteps": args.timesteps,
                "Initial Conv Dim": args.initial_dim,
                "Batch Size": args.batch_size,
                "n_fft": args.n_fft,
                "n_mels": args.n_mels,
                "hop_length": args.hop_length,
                "depth": args.depth
            }
        )
        wandb.watch(ddpm)

    models_saved = 0
    best_loss = float("inf")
    for epoch in range(args.epochs):
        ddpm.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}")

        for i, examples in enumerate(pbar):
            optimizer.zero_grad()
            noise = torch.randn_like(examples).to(device)
            t = torch.randint(0, args.timesteps, [len(examples)]).to(device)

            noisy_examples = ddpm(examples, t, noise)
            pred_noise = ddpm.backward(noisy_examples, t)

            loss = loss_fn(pred_noise, noise)
            loss.backward()

            optimizer.step()

            loss_item = loss.detach().item()
            
            logs = {"loss": loss_item, "models_saved": models_saved}
            pbar.set_postfix(**logs)

            if args.wandb:
                wandb.log(logs)
        
        if loss_item < best_loss:
            best_loss = loss_item
            if not os.path.exists("models/"):
                os.mkdir("models/")
            torch.save(ddpm.state_dict(), "models/model.pt")
            models_saved += 1

        if (epoch + 1) % args.epochs_pred == 0:
            ddpm.eval()
            samples = ddpm.sample(1, args.n_mels, args.seq_length)
            sample = samples[-1].cpu().detach().numpy().squeeze()

            if args.wandb:
                image = wandb.Image(sample)
                wandb.log({"Prediction": image})

if __name__ == "__main__":
    main()