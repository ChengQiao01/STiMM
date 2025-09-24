from transformers import ViTConfig


# 创建一个继承 ViTConfig 的子类
class TimeSeriesFormeConfig(ViTConfig):
    def __init__(
            self,
            time_num=128,
            trace_num=128,
            **kwargs):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
        # 新增的自定义变量
        self.time_num = time_num
        self.trace_num = trace_num


