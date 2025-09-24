import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from scripts.scheduler import CycleScheduler
from scripts.DataLoad_Train import DataLoad_Train
from scripts.trainer import train_and_validate
from src.STiMM.Time_Seriosformer_v1_2 import Seis_Time_Transformer
from src.STiMM.Time_Series_Config import TimeSeriesFormeConfig


def main(args, model, SeisTrace_config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DataLoad_Train(args.data_dir, "dataset_all", data_type='v7', chanell_dim=[0, 2, 1],
                                     expand_dim=1)

    train_size = int(args.train_rate * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.BatchSize, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.BatchSize, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler_train = None
    if args.sched == "cycle":
        scheduler_train = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(train_loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )
    train_and_validate(model=model, train_loader=train_loader, val_loader=val_loader,
                       optimizer=optimizer, scheduler=scheduler_train,
                       config=SeisTrace_config, args=args,
                       device=device,
                       epochs=args.epoch,
                       mask_ratio=args.mask_rate,
                       save_last=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--BatchSize", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--mask_rate", type=float, default=0.75)
    parser.add_argument("--net_version", type=str, default="Seis_TimeFormer_v1_5")
    parser.add_argument("--loss_type", type=str, default="L1Loss")
    parser.add_argument("--sched", type=str, default="cycle")
    parser.add_argument("--data_dir", type=str,
                        default='/mnt/data/chengqiao427/b_code_python/STT/dataset/train_data/dataset.mat')
    parser.add_argument("--data_version", type=str, default='dataset_all')
    parser.add_argument("--save_epoch", type=float, default=100)
    parser.add_argument("--train_rate", type=float, default=0.7)
    args = parser.parse_args()

    SeisTrace_config = TimeSeriesFormeConfig(time_num=2048, trace_num=128, num_channels=1, patch_size=4,
                                            hidden_size=512, intermediate_size=2048, num_attention_heads=8,
                                            num_hidden_layers=12)
    model = Seis_Time_Transformer(SeisTrace_config)
    main(args, model, SeisTrace_config)
