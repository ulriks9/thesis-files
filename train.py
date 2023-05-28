import torch
import argparse
import wandb
import os
from datasets import FMA
from unet import UNet1D
from ddpm import DDPM
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser(
    prog="1D DDPM Train Script"
)

parser.add_argument('--wandb', required=False, type=str, default='True')
parser.add_argument('--epochs', default=2000, required=False, type=int)
parser.add_argument('--epochs_pred', default=5, required=False, type=int)
parser.add_argument('--timesteps', default=4000, required=False, type=int)
parser.add_argument('--batch_size', default=32, required=False, type=int)
parser.add_argument('--n_fft', default=4096, required=False, type=int)
parser.add_argument('--n_mels', default=256, required=False, type=int)
parser.add_argument('--hop_length', default=2585, required=False, type=int)
parser.add_argument('--time_embedding_dim', default=256, required=False, type=int)
parser.add_argument('--seq_length', default=512, required=False, type=int)
parser.add_argument('--verbose', default='True', required=False, type=str)
parser.add_argument('--optimizer', default="AdamW", required=False, type=str)
parser.add_argument('--l_r', default=1e-4, required=False, type=float)
parser.add_argument('--eps', default=1e-6, required=False, type=float)
parser.add_argument('--weight_decay', default=1e-3, required=False, type=float)
parser.add_argument('--num_workers', default=cpu_count() - 2, required=False, type=int)
parser.add_argument('--dims', default="256,256,512,512,1024,1024", required=False, type=str)
parser.add_argument('--scheduler', default="ReduceLROnPlateau", required=False, type=str)
parser.add_argument('--patience', default=10, required=False, type=int)
parser.add_argument('--scheduler_threshold', default=1e-2, required=False, type=int)

def main():
    device = "cuda:0"
    args = parser.parse_args()
    args.dims = str(args.dims).split(',')
    args.dims = [int(dim) for dim in args.dims]
    dataset = FMA(workers=args.num_workers, sample_shape=(args.n_mels, args.seq_length))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    loss_fn = torch.nn.MSELoss()

    # Argparse doesn't convert to bool:
    if args.wandb in 'True':
        WANDB = True
    else:
        WANDB = False
    
    if args.verbose in 'True':
        args.verbose = True
    else:
        args.verbose = False

    model = UNet1D(
        in_channels=args.n_mels,
        out_channels=args.n_mels,
        dims=args.dims,
        time_embedding_dim=args.time_embedding_dim,
        device=device
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
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=args.l_r, eps=args.eps, weight_decay=args.weight_decay)
    elif args.optimizer == "RMSProp":
        optimizer = torch.optim.RMSprop(ddpm.parameters(), lr=args.l_r)
    elif args.optimizer == "SGD":
        optimizer = torch.optim.SGD(ddpm.parameters(), lr=args.l_r)

    scheduler = ReduceLROnPlateau(
        optimizer, 
        patience=args.patience,
        threshold=args.scheduler_threshold
        )

    if WANDB == True:
        wandb.login()
        wandb.init(
            project="confident-diffusion",
            config={
                "Epochs": args.epochs,
                "Timesteps": args.timesteps,
                "Batch Size": args.batch_size,
                "Number of FFTs": args.n_fft,
                "Number of Mels": args.n_mels,
                "Hop Length": args.hop_length,
                "Optimizer": args.optimizer,
                "Learning Rate": args.l_r,
                "Patience": args.patience
            }
        )

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
            
            logs = {
                "Loss": loss_item, 
                "Number of Models Saved": models_saved,
                "Current Learning Rate": optimizer.param_groups[0]['lr']
                }
            
            pbar.set_postfix(**logs)

            if WANDB == True:
                wandb.log(logs)
        
        scheduler.step(loss_item)

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

            if WANDB == True:
                image = wandb.Image(sample)
                wandb.log({"Prediction": image})
            
            del(samples)

if __name__ == "__main__":
    main()