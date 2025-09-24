from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def is_integer_ratio(x):
    return math.isclose(x, round(x), rel_tol=1e-6)


class TimeUpBlock(nn.Module):
    def __init__(self, factor, target_channels):
        super(TimeUpBlock, self).__init__()
        self.factor = factor
        self.expand = nn.Conv2d(1, factor, kernel_size=1, bias=False)
        self.smooth = nn.Conv2d(1, target_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.expand(x)                  # [B, factor, T, X]
        B, C, T, X = x.shape
        x = x.permute(0, 2, 1, 3)           # [B, T, factor, X]
        x = x.reshape(B, 1, T * self.factor, X)  # [B, 1, T*factor, X]
        x = self.smooth(x)                  # [B, target_channels, T*factor, X]
        return x


class SeisTimeResizer(nn.Module):
    def __init__(self, target_dt, target_channels):
        super(SeisTimeResizer, self).__init__()
        self.target_dt = target_dt
        self.target_channels = target_channels
        self.mapper = nn.Conv2d(1, target_channels, kernel_size=1)

    def forward(self, x, input_dt):
        """
        输入：
            x: Tensor [B, 1, T, X]
            input_dt: Tensor [B, 1]，表示每个样本的时间采样间隔
        输出：
            Tensor [B, target_channels, T', X]
        """
        B, _, T, X = x.shape
        out_list = []

        for i in range(B):
            xi = x[i:i+1]                  # [1, 1, T, X]
            dt = input_dt[i, 0].item()     # scalar

            if is_integer_ratio(self.target_dt / dt):
                # 下采样：采样率变小，时间维减少
                factor = int(round(self.target_dt / dt))
                conv = nn.Conv2d(1, self.target_channels, kernel_size=(factor, 1),
                                 stride=(factor, 1), bias=False).to(x.device)
                out = conv(xi)

            elif is_integer_ratio(dt / self.target_dt):
                # 上采样：采样率变大，时间维增加
                factor = int(round(dt / self.target_dt))
                up = TimeUpBlock(factor, self.target_channels).to(x.device)
                out = up(xi)

            else:
                # fallback 插值
                T_target = int(round(dt * T / self.target_dt))
                out = F.interpolate(xi, size=(T_target, X), mode='bilinear', align_corners=True)
                out = self.mapper.to(x.device)(out)

            out_list.append(out)

        return torch.cat(out_list, dim=0)  # 合并所有样本


class SpaceUpBlock(nn.Module):
    def __init__(self, factor, target_channels):
        super(SpaceUpBlock, self).__init__()
        self.factor = factor
        self.expand = nn.Conv2d(target_channels, factor * target_channels, kernel_size=1, bias=False)
        self.smooth = nn.Conv2d(target_channels, target_channels, kernel_size=3, padding=1)

    def forward(self, x):
        B, C, T, X = x.shape
        x = self.expand(x)                  # [B, factor, T, X]
        x = x.view(B, self.factor, C, T, X)
        x = x.permute(0, 2, 3, 4, 1)  # [B, C, T, X, factor]
        x = x.reshape(B, C, T, X * self.factor)  # [B, 1, T, X*factor]
        x = self.smooth(x)                  # [B, target_channels, T, X*factor]
        return x

class SeisSpaceResizer(nn.Module):
    def __init__(self, input_dx, target_dx, target_channels):
        super(SeisSpaceResizer, self).__init__()
        self.input_dx = input_dx
        self.target_dx = target_dx
        self.target_channels = target_channels

        self.mode = None
        self.factor = None

        # 默认值（在 forward 中根据采样率自动构建）
        self.down = None
        self.up_expand = None
        self.mapper = None
        self._build_mode()

    def _build_mode(self):
        if is_integer_ratio(self.target_dx / self.input_dx):
            self.mode = 'down'
            self.factor = int(round(self.target_dx / self.input_dx))
            self.down = nn.Conv2d(self.target_channels, self.target_channels, kernel_size=(1, self.factor),
                                  stride=(1, self.factor), bias=False)
        elif is_integer_ratio(self.input_dx / self.target_dx):
            self.mode = 'up'
            self.factor = int(round(self.input_dx / self.target_dx))
            self.up_expand = SpaceUpBlock(self.factor, self.target_channels)
        else:
            self.mode = 'interp'
            self.mapper = nn.Conv2d(self.target_channels, self.target_channels, kernel_size=1)

    def forward(self, x):
        """
        输入：x, shape = [B, C, T, X]
        输出：空间维已重采样的张量 [B, C, T, X']
        """
        B, C, T, X = x.shape

        if self.mode == 'down':
            return self.down(x)

        elif self.mode == 'up':
            return self.up_expand(x)                # [B, target_channels, T*factor, X]

        else:
            # fallback: interpolate 到目标尺寸
            target_X = int(round(self.input_dx * X / self.target_dx))
            x = F.interpolate(x, size=(T, target_X), mode='bilinear', align_corners=True)

            return self.mapper(x)


class ResizeAndProject(nn.Module):
    def __init__(self, out_channels=1, target_size=(128, 128)):
        super(ResizeAndProject, self).__init__()
        self.target_size = target_size  # (m, n)
        self.project = nn.Conv2d(1, out_channels, kernel_size=1)
        self.out_project = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        x: Tensor of shape [m_i, n_i] (not batched)
        Returns: Tensor of shape [c, m, n]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif x.dim() == 3:
            x = x.unsqueeze(0)  # [1, 1, H, W]
        elif x.dim() == 4:
            pass  # [B, 1, H, W]
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
        # Step 1: resize 到统一尺寸
        x = self.project(x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        # Step 2: 1×1 conv 升通道 → [1, c, m, n]
        x = self.out_project(x)

        return x.squeeze(0)  # 返回 [c, m, n]


class SeisResizeAndProject(nn.Module):
    def __init__(self, out_channels=1, target_size=(128, 128), target_interval=(2, 10)):
        super(SeisResizeAndProject, self).__init__()
        self.target_size = target_size  # (m, n)
        self.target_interval = target_interval
        self.target_dt = target_interval[0]
        self.target_dx = target_interval[1]
        self.out_channels = out_channels
        self.project = nn.Conv2d(1, out_channels, kernel_size=1)
        self.out_project = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        self.Time_resize = SeisTimeResizer()
        self.Space_resize = SpaceUpBlock()

    def forward(self, x, input_interval):
        """
        x: Tensor of shape [m_i, n_i] (not batched)
        Returns: Tensor of shape [c, m, n]
        """
        [b, c, nt, nx] = x.shape()
        input_dt = input_interval[0]
        input_dx = input_interval[1]

        x = seis_time_resize(x, input_dt, nt, nx, self.target_dt, self.target_channels)


        # Step 1: resize 到统一尺寸
        x = self.project(x)
        x = F.interpolate(x, size=self.target_size, mode='bilinear', align_corners=True)
        # Step 2: 1×1 conv 升通道 → [1, c, m, n]
        x = self.out_project(x)

        return x.squeeze(0)  # 返回 [c, m, n]

class Time_Series_PatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, time_num, trace_num)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        # input data parameter
        self.time_num = config.time_num
        self.trace_num = config.trace_num

        # trans input data physical time-space to traget time-space
        self.target_seis_dx = config.target_seis_dx
        self.target_seis_dt = config.target_seis_dt
        self.target_seis_nx = config.target_seis_nx
        self.target_seis_nt = config.target_seis_nt
        self.time_window = config.time_window

        target_interval = [self.target_seis_dt, self.target_seis_dx]
        target_size = [self.time_window, self.target_seis_nx]
        self.projection = ResizeAndProject(out_channels=self.target_features, )

        #

        self.time_win_split_step = config.time_win_split_step
        self.target_features = config.target_features


        hidden_size = config.hidden_size
        self.num_channels = config.num_channels
        self.num_patches = self.trace_num


        # self.projection = nn.Conv2d(self.num_channels, hidden_size, kernel_size=[self.time_num, 1], stride=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape

        if height != self.time_num or width != self.trace_num:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model"
                f" ({self.time_num}*{self.trace_num})."
            )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings

# 测试一下模型是否正常运行
if __name__ == "__main__":
    # 模拟三个不同大小的输入
    input1 = torch.randn(10, 1, 100, 150)
    input2 = torch.randn(10, 1, 64, 64)
    input3 = torch.randn(10, 1, 200, 120)

    # 模块初始化（设输出通道为 3，目标空间大小为 128×128）
    model = ResizeAndProject(out_channels=3, target_size=(128, 128))

    # 应用到每个输入上
    outputs = [model(x) for x in [input1, input2, input3]]

    # 打印结果维度
    for i, out in enumerate(outputs):
        print(f"Output {i+1} shape: {out.shape}")
    # device = "cuda"
    # x = torch.randn(2, 1, 32, 128).to(device)
    # resizer = SeisTimeResizer(input_dt=3, target_dt=2, target_channels=8).to(device)
    # y = resizer(x)
    # print(y.shape)
    # x_resizer = SeisSpaceResizer(input_dx=25, target_dx=20, target_channels=8).to(device)
    # z = x_resizer(y)
    # print(z.shape)
    # x = torch.randn(3, 1, 64, 128)  # [B=3, 1, T=64, X=128]
    # input_dt = torch.tensor([[2.], [4.], [2.5]])  # [3, 1]
    # resizer = SeisTimeResizer(target_dt=1.0, target_channels=8)
    #
    # output = resizer(x, input_dt)
    # print(output.shape)  # 输出形状依采样比而定，时间维不同，取决于每个样本

