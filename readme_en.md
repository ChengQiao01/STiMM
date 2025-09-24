# STiMM — Seismic Time-series Masked Modeling
> A Transformer-based framework for **seismic time-series** masked modeling and reconstruction (Seismic Time-series Masked Modeling, **STiMM**).  
> **Out-of-the-box**: includes sample data `data/test_dataset.mat` and pretrained weights `models/ckpt_STiMM.pth`.

![Framework of STiMM](docs/method.png)

## Project Structure
STiMM/
├─ data/
│ └─ test_dataset.mat # Example dataset
├─ models/
│ └─ ckpt_STiMM.pth # Example / pretrained weights
├─ results/
│ └─ Loss/ # Training curves / logs (auto-generated)
├─ scripts/
│ ├─ activations_hf.py
│ ├─ DataLoad_Train.py # Data loading & mask generation
│ ├─ PixelShuffle1D.py
│ ├─ scheduler.py
│ ├─ tester.py
│ └─ trainer.py
├─ src/
│ ├─ STiMM/
│ │ ├─ readme
│ │ ├─ Time_Series_Config.py # Global configuration (main place to modify)
│ │ ├─ Time_Series_Embedding.py
│ │ ├─ Time_Seriosformer_v1_1.py
│ │ └─ Time_Seriosformer_v1_2.py
│ └─ ViT/
│ ├─ config_vit_hf.py
│ └─ ViT_seis_emb.py
├─ Train_Seis_Time_Former.py # Training entry
├─ Test_Seis_Time_Former.py # Testing / inference entry
├─ requirements.txt
└─ readme.md

---

## Installation
```bash
conda create -n stimm python=3.11 -y
conda activate stimm
pip install -r requirements.txt
```

## Data
-**Example dataset**: data/test_dataset.mat is provided with the repository.

-**Custom data**: please refer to the logic in scripts/DataLoad_Train.py and adjust the variable/key names.

-**Convention**:
 -Model input tensor shape: (B, 1, T, X), where T is the time dimension, X is the spatial dimension.

## Quick Start
With the pretrained model downloaded, you can directly run:
```bash
python Test_Seis_Time_Former.py
```
to see pretrained reconstruction results.

### Training
Run with the default config and sample dataset:
```bash
python Train_Seis_Time_Former.py
```

### Testing
Run with sample weights and dataset:
```bash
python Test_Seis_Time_Former.py
```
The script will call scripts/tester.py for evaluation and visualization, saving outputs to results/.
> If you replace weights/data, please modify paths in the script or adjust in the config file.
> Current version requires input shape [B, 1, 2048, 128].

---

## Configuration
Centralized in 'src/STiMM/Time_Series_Config.py':

---

## Models & Releases
- Pretrained model downloadhttps://github.com/ChengQiao01/STiMM/releases/download/v0.1.1/ckpt_STiMM.pth
- Place the downloaded weights into the '/models/' directory.

---
