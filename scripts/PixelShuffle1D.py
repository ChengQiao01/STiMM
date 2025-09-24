import torch
import torch.nn as nn


class PixelShuffle1D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        batch_size, channels_in, T, X = input.size()
        r = self.upscale_factor
        channels_out = channels_in // r
        assert channels_in % r == 0, '通道数必须能被 upscale_factor 整除'

        # 重新调整张量的形状
        input = input.contiguous().view(batch_size, channels_out, r, T, X)

        # 重新排列张量的维度
        input = input.permute(0, 1, 3, 4, 2).contiguous()

        # 合并维度以得到最终的输出形状
        output = input.view(batch_size, channels_out, T, X * r)
        return output


class PixelShuffle1D_Time(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1D_Time, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        batch_size, channels_in, T, X = input.size()
        r = self.upscale_factor
        channels_out = channels_in // r
        assert channels_in % r == 0, '通道数必须能被 upscale_factor 整除'

        # 重新调整张量的形状
        input = input.contiguous().view(batch_size, channels_out, r, T, X)

        # 重新排列张量的维度
        input = input.permute(0, 1, 3, 2, 4).contiguous()

        # 合并维度以得到最终的输出形状
        output = input.view(batch_size, channels_out, T * r, X)

        return output
# 测试一下模型是否正常运行
if __name__ == "__main__":
    model = PixelShuffle1D_Time(upscale_factor=2)
    input_image = torch.randn(1, 2, 256, 256)  # batch size=1, channel=1, H=256, W=256
    input_tensor = torch.tensor([[[[1, 2, 3],
                                   [4, 5, 6],
                                   [7, 8, 9],
                                   [10, 11, 12]]]], dtype=torch.float).view(1, 2, 2, 3)
    output_image = model(input_image)
    print(input_tensor)
    print(output_image)  # 应该是 torch.Size([1, 1, 256, 256])