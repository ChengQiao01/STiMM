import torch
import scipy

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from scripts.DataLoad_Train import generate_random_bool_masked_pos


def _expand_patch_mask(bool_masked_pos: torch.Tensor, patch_size: int, T: int) -> torch.Tensor:
    """
    将 [B, P] 的patch级bool掩码扩展为 [B, T] 的时间级bool掩码（最后一维为时间）。
    """
    # 重复每个patch的标记到其对应的 patch_size 个时间采样点
    m = bool_masked_pos.repeat_interleave(patch_size, dim=1)
    if m.size(1) < T:
        # 安全兜底（极少见）：补到T
        pad = T - m.size(1)
        m = torch.cat([m, m[:, -1:].expand(-1, pad)], dim=1)
    return m[:, :T]  # [B, T]


def _apply_temporal_mask(pixel_values: torch.Tensor, mask_T: torch.Tensor, fill: float) -> torch.Tensor:
    """
    将 [B, T] 的时间掩码广播到 x 的形状 [B, ..., T]，并以 fill 值填充被掩码位置。
    """
    B, C, T, X = pixel_values.shape
    assert mask_T.shape == (B, T)
    mask_T = mask_T.unsqueeze(1)
    mask_b = mask_T.unsqueeze(-1).repeat(1, 1, 1, X)
    return pixel_values.masked_fill(mask_b, fill)


@torch.no_grad()
def test(
        model,
        test_loader: DataLoader,
        config,
        results_save_dir_mat=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        mask_ratio=0.5,
        show_figure=None
):
    model.to(device).eval()
    pre = []
    Sample = []
    Label = []

    test_loss = 0.0
    for batch in tqdm(test_loader, desc="Test"):
        pixel_values = batch.to(device)
        B, _, T, _ = pixel_values.shape
        bool_masked_pos = generate_random_bool_masked_pos(
            batch_size=pixel_values.size(0),
            num_patches=config.time_num // config.patch_size,
            mask_ratio=mask_ratio,
        ).to(device)

        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
        test_loss += outputs.loss.item()
        out = outputs.reconstruction
        pre.append(out.cpu().detach().numpy())
        Label.append(pixel_values.cpu().detach().numpy())

        mask_pos = _expand_patch_mask(bool_masked_pos, config.patch_size, T)
        x_masked = _apply_temporal_mask(pixel_values, mask_pos, fill=0)
        Sample.append(x_masked.cpu().detach().numpy())
    pre = np.concatenate(pre, axis=0)
    Sample = np.concatenate(Sample, axis=0)
    Label = np.concatenate(Label, axis=0)

    avg_val_loss = test_loss / len(test_loader)
    print(f"Test Avg Loss: {avg_val_loss:.6f}")

    if results_save_dir_mat is not None:
        # ======= 测试结果保存为.mat =======
        data_out = {
            'pre': pre,
            'sample': Sample,
            'label': Label,
        }
        scipy.io.savemat(results_save_dir_mat, data_out)
    if show_figure:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, data, title in zip(
                axes,
                [Sample[0, 0], Label[0, 0], pre[0, 0]],
                ['Masked Input', 'Raw Seismic', 'Output']
        ):
            im = ax.imshow(data, cmap='seismic', aspect='auto', vmin=-0.2, vmax=0.2)
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
