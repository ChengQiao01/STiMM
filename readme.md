# STiMM — Seismic Time-series Masked Modeling
> 基于 Transformer 的**地震时间序列**掩码建模/重建框架（Seismic Time-series Masked Modeling, **STiMM**）。  
> **开箱即用**：含示例数据 `data/test_dataset.mat` 与示例权重 `models/ckpt_STiMM.pth`。


## 目录结构
```
STiMM/
├─ data/
│  └─ test_dataset.mat              # 示例数据
├─ models/
│  └─ ckpt_STiMM.pth                # 示例/预训练权重
├─ results/
│  └─ Loss/                         # 训练曲线/日志（脚本生成）
├─ scripts/
│  ├─ activations_hf.py
│  ├─ DataLoad_Train.py             # 数据加载 & 掩码生成
│  ├─ PixelShuffle1D.py
│  ├─ scheduler.py
│  ├─ tester.py
│  └─ trainer.py
├─ src/
│  ├─ STiMM/
│  │  ├─ readme
│  │  ├─ Time_Series_Config.py      # 全局配置（建议主要在此改）
│  │  ├─ Time_Series_Embedding.py
│  │  ├─ Time_Seriosformer_v1_1.py
│  │  └─ Time_Seriosformer_v1_2.py
│  └─ ViT/
│     ├─ config_vit_hf.py
│     └─ ViT_seis_emb.py
├─ Train_Seis_Time_Former.py        # 训练入口
├─ Test_Seis_Time_Former.py         # 测试/推理入口
├─ requirements.txt
└─ readme.md
```

---

## 安装（Install）
```bash
conda create -n stimm python=3.11 -y
conda activate stimm
pip install -r requirements.txt
```

---

## 数据（Data）
- **示例数据**：`data/test_dataset.mat` 已随仓库提供，用于跑通流程。  
- **自有数据**：请参考 `scripts/DataLoad_Train.py` 的读取逻辑修改变量名/键值。  
- **约定**：
  - 张量形状为 `(..., T)`，**最后一维为时间 `T`**；
  - `T % patch_size == 0`（沿 `T` 切 patch），相关参数见 `src/STiMM/Time_Series_Config.py`；
  - 若是二维剖面常见为 `(H, T)` 或 `(C, T)`；三维体可先在 DataLoader 中折叠/展平到以 `T` 为最后一维的形态。

---

## 快速开始（Quick Start）

### 训练
使用仓库内默认配置与示例数据：
```bash
python Train_Seis_Time_Former.py
```
> 常用超参（`time_num / patch_size / mask_ratio / batch_size / lr / epochs` 等）在  
> `src/STiMM/Time_Series_Config.py` 中集中管理。

### 测试 / 推理
使用示例权重与示例数据：
```bash
python Test_Seis_Time_Former.py
```
脚本会调用 `scripts/tester.py` 进行评测与可视化，输出保存到 `results/`。  
> 若你替换权重/数据，请按脚本注释修改路径或在配置中调整。

---

## 配置（Config）
统一在 `src/STiMM/Time_Series_Config.py` 维护：
- **数据与 I/O**：数据根目录、输出目录、训练/验证/测试拆分；
- **模型结构**：`embed_dim / depth / num_heads / drop_path / act`；
- **掩码策略**：`patch_size / mask_ratio / sampler`；
- **优化与调度**：`optimizer / lr / weight_decay / warmup / scheduler`；
ViT 风格嵌入的附加参数位于 `src/ViT/config_vit_hf.py`。

---

## 模型与发布（Models & Releases）
- 预训练的模型下载 https://github.com/ChengQiao01/STiMM/releases/download/v0.1.1/ckpt_STiMM.pth
- 将下载模型放置在 '/models/' 中

---
