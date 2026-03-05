# NOMA_AMC_MRCNN

> **Base Repository:** Supplementary material for the paper **"Modulation Classification for Non-orthogonal Multiple Access System using a Modified Residual-CNN"**
> 📄 [https://doi.org/10.1109/WCNC55385.2023.10118621](https://doi.org/10.1109/WCNC55385.2023.10118621)
>
> **Extended & Maintained by:** Priyanshu Swain, Lalit Mohan
> *As part of the Attentive HybNet for NOMA-AMC research project*

---

## 📌 What This Repository Contains

| File | Description |
|---|---|
| `NOMA_AMC_dataset_generation.m` | MATLAB script to generate the NOMA dataset |
| `data_loaders.py` | Python script to load, verify and visualize the dataset |
| `model_weights/` | Pre-trained model weights from the original paper |

---

## 📡 Dataset Generation (MATLAB)

The MATLAB file generates a simulated NOMA dataset with Rayleigh fading and AWGN noise.

### Key Parameters

| Parameter | Description | Default |
|---|---|---|
| `N` | Signal length (number of symbols per sample) | `200` |
| `Pf` | Power allocated to the **far** user | `0.8` |
| `Pn` | Power allocated to the **near** user | `0.2` |
| `vriance` | Rayleigh fading variance | `0.1` |
| SNR Range | -10 dB to +20 dB in steps of 2 dB | 16 levels |
| Samples | 1000 per SNR level | 16,000 total |

### Modulation Classes

| Label | Modulation |
|---|---|
| 0 | BPSK |
| 1 | QPSK |
| 2 | 8-PSK |
| 3 | 16-QAM |

### How to Run

1. Open `NOMA_AMC_dataset_generation.m` in MATLAB
2. Set your desired `N` and power values at the top
3. Run the script — a progress bar will show overall % completion
4. Output saved as `myfile.mat` (v7.3 format, supports >2GB)

> ⚠️ **Note from original author:** There is a typo in the paper's model figure — kernel size in the second convolution layer should be **(2×8)**, not (2×4).

---

## 🐍 Loading & Visualizing in Python

### Requirements
```bash
pip install numpy h5py matplotlib
```

### Run
```bash
python data_loaders.py
```

### What it does
- Loads `myfile.mat` and prints a full dataset summary
- Fixes the MATLAB → Python complex number format automatically
- Generates **3 diagnostic plots** and saves them as `.png` files

| Plot | What you see |
|---|---|
| `plot1_constellations.png` | All 4 modulations at low (-10 dB) vs high (+20 dB) SNR |
| `plot2_snr_progression.png` | One modulation across all 16 SNR levels |
| `plot3_statistics.png` | Class distribution, SNR distribution, raw I/Q signal |

### Expected Output
```
Data shape    : (16000, 2, 200)
Labels shape  : (16000,)
SNRs shape    : (16000,)
Unique labels : [0 1 2 3]  → ['BPSK', 'QPSK', '8PSK', '16QAM']
Unique SNRs   : [-10 -8 -6 -4 -2 0 2 4 6 8 10 12 14 16 18 20] dB
```

---

## 🏋️ Pre-trained Model Weights

The `model_weights/` folder contains weights for all parameter combinations from the original paper. Load them to reproduce paper results without retraining.

| Weight File | N | Power Ratio |
|---|---|---|
| `NOMA_model7` | 200 | 2 |
| `NOMA_model7_100_4` | 100 | 4 |
| `NOMA_model7_200_2_lowLR` | 200 | 2 (low LR) |
| `NOMA_model7_400_4` | 400 | 4 |
| `NOMA_model7_800_2` | 800 | 2 |
| `NOMA_model7_800_4` | 800 | 4 |
| `NOMA_model7_N_800_1_2` | 800 | 1.2 |
| `NOMA_model7_N_800_1_5` | 800 | 1.5 |

---

## 🔗 Original Dataset (Pre-generated)

The original author generated signals for **power ratio = 4, N = 200** in pickle format.
Download from here: [Google Drive Link](https://drive.google.com/file/d/1nFIllsllFieRZaGeZTHFxiiXaF7ht4vQ/view?usp=sharing)

---

## 🚀 How This Repo Fits Into Our Project

This repository serves as the **verified data generation baseline** for our larger project:

> **Attentive HybNet** — a multi-modal deep learning architecture that combines:
> - A **Temporal Stream** (raw I/Q → TCN + Channel Attention)
> - A **Visual Stream** (KDE map → CNN + Spatial Attention)
> - **Attention-based fusion** for final classification

The dataset generated here feeds directly into our Attentive HybNet pipeline.

---

## 📬 Contact

- **Original paper author:** Ashok Parmar
- **This fork & extensions:** Priyanshu Swain, Lalit Mohan
