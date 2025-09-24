import argparse
import torch
from torch.utils.data import DataLoader
from scripts.DataLoad_Train import DataLoad_Train
from scripts.tester import test
from src.STiMM.Time_Seriosformer_v1_2 import Seis_Time_Transformer
from src.STiMM.Time_Series_Config import TimeSeriesFormeConfig


def main(args, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    dataset = DataLoad_Train(args.data_dir, "dataset", data_type='v7', chanell_dim=[0, 2, 1],
                                     expand_dim=1)
    # dataset = torch.randn([32, 1, 2048, 128])
    test_loader = DataLoader(dataset, batch_size=args.BatchSize, shuffle=False, num_workers=4, pin_memory=True)
    test(model=model, test_loader=test_loader,
         config=SeisTrace_config,
         device=device,
         mask_ratio=args.mask_rate,
         show_figure=args.plot_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--BatchSize", type=int, default=32)
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--data_dir", type=str,
                        default='./data/test_dataset.mat')
    parser.add_argument("--plot_results", type=int, default=1)
    args = parser.parse_args()

    models_save_dir = './models/ckpt_STiMM.pth'
    SeisTrace_config = TimeSeriesFormeConfig(time_num=2048, trace_num=128, num_channels=1, patch_size=4,
                                             hidden_size=512, intermediate_size=2048, num_attention_heads=8,
                                             num_hidden_layers=12)
    model = Seis_Time_Transformer(SeisTrace_config)
    model.load_state_dict(torch.load(models_save_dir, weights_only=True))
    main(args, model)