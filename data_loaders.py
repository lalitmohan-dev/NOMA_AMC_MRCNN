# ============================================================
# NOMA-AMC Dataset Loader & Visualizer
# Authors: Priyanshu Swain, Lalit Mohan
# ============================================================

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MAT_FILE    = "myfile_ratio2.mat"
CLASS_NAMES = ['BPSK', 'QPSK', '8PSK', '16QAM']
SNR_LEVELS  = list(range(-10, 22, 2))   # -10 to +20 step 2

# ─────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────
print("=" * 50)
print("  NOMA-AMC Dataset Loader")
print("=" * 50)
print(f"\n Loading: {MAT_FILE} ...")

with h5py.File(MAT_FILE, 'r') as f:
    print(f" Keys found : {list(f.keys())}")
    X_raw  = np.array(f['data_Y'])
    labels = np.array(f['true_Mods']).flatten()
    snrs   = np.array(f['snrs']).flatten()

# Convert MATLAB complex struct → Python complex, fix axis order
X_complex = (X_raw['real'] + 1j * X_raw['imag']).T   # (16000, 200)

# Build standard Xs array  (samples × 2 × signal_length)
Xs        = np.zeros((X_complex.shape[0], 2, X_complex.shape[1]))
Xs[:, 0, :] = X_complex.real
Xs[:, 1, :] = X_complex.imag

# ─────────────────────────────────────────
# STEP 2 — PRINT SUMMARY
# ─────────────────────────────────────────
print("\n── Dataset Summary ──────────────────────")
print(f"  Xs shape      : {Xs.shape}   (samples, I/Q, length)")
print(f"  Labels shape  : {labels.shape}")
print(f"  SNRs shape    : {snrs.shape}")
print(f"  Unique labels : {np.unique(labels).astype(int)}  → {CLASS_NAMES}")
print(f"  Unique SNRs   : {np.unique(snrs).astype(int)} dB")
print(f"  Total samples : {len(labels)}")
print("─" * 42)
for idx, name in enumerate(CLASS_NAMES):
    count = np.sum(labels == idx)
    print(f"  {name:8s} : {count:5d} samples ({100*count/len(labels):.1f}%)")
print("=" * 50)


# ─────────────────────────────────────────
# HELPER — get one sample cleanly
# ─────────────────────────────────────────
def get_sample(mod_idx, snr_db):
    """Return first sample matching modulation class and SNR."""
    mask = (labels == mod_idx) & (snrs == snr_db)
    idx  = np.where(mask)[0]
    if len(idx) == 0:
        return None
    return X_complex[idx[0]]


# ─────────────────────────────────────────
# PLOT 1 — All 4 modulations at LOW vs HIGH SNR
# ─────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Constellation Diagrams — All Modulations @ Low vs High SNR",
             fontsize=14, fontweight='bold')

for col, name in enumerate(CLASS_NAMES):
    for row, (snr_val, snr_label) in enumerate([(-10, "SNR = -10 dB (Noisy)"),
                                                 ( 20, "SNR = +20 dB (Clean)")]):
        sample = get_sample(col, snr_val)
        ax     = axes[row][col]

        if sample is not None:
            ax.scatter(sample.real, sample.imag,
                       s=2, alpha=0.4, color='steelblue')
        ax.set_title(f"{name}\n{snr_label}", fontsize=9)
        ax.set_xlabel("I (In-phase)",    fontsize=7)
        ax.set_ylabel("Q (Quadrature)",  fontsize=7)
        ax.grid(True, linewidth=0.4)
        ax.axis('equal')
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig("plot1_constellations.png", dpi=150)
plt.show()
print(" Saved → plot1_constellations.png")


# ─────────────────────────────────────────
# PLOT 2 — One modulation across ALL SNR levels
# ─────────────────────────────────────────
MOD_TO_SHOW = 1   # 0=BPSK, 1=QPSK, 2=8PSK, 3=16QAM

fig, axes = plt.subplots(2, 8, figsize=(22, 6))
fig.suptitle(f"Constellation of {CLASS_NAMES[MOD_TO_SHOW]} Across All SNR Levels",
             fontsize=13, fontweight='bold')

for idx, snr_val in enumerate(SNR_LEVELS):
    row = idx // 8
    col = idx  % 8
    ax  = axes[row][col]
    sample = get_sample(MOD_TO_SHOW, snr_val)
    if sample is not None:
        ax.scatter(sample.real, sample.imag,
                   s=2, alpha=0.5,
                   color='tomato' if snr_val < 0 else
                         'orange' if snr_val < 10 else 'green')
    ax.set_title(f"{snr_val} dB", fontsize=8)
    ax.axis('equal')
    ax.grid(True, linewidth=0.3)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
plt.savefig("plot2_snr_progression.png", dpi=150)
plt.show()
print(" Saved → plot2_snr_progression.png")


# ─────────────────────────────────────────
# PLOT 3 — Dataset Statistics
# ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Dataset Statistics", fontsize=13, fontweight='bold')

# (a) Class distribution bar chart
class_counts = [np.sum(labels == i) for i in range(4)]
bars = axes[0].bar(CLASS_NAMES, class_counts,
                   color=['#4e79a7','#f28e2b','#e15759','#76b7b2'],
                   edgecolor='black', linewidth=0.7)
axes[0].set_title("Samples per Modulation Class")
axes[0].set_ylabel("Count")
axes[0].set_ylim(0, max(class_counts) * 1.2)
for bar, count in zip(bars, class_counts):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 30,
                 str(count), ha='center', fontsize=9)

# (b) SNR distribution
snr_counts = [np.sum(snrs == s) for s in SNR_LEVELS]
axes[1].bar(SNR_LEVELS, snr_counts, width=1.5,
            color='steelblue', edgecolor='black', linewidth=0.5)
axes[1].set_title("Samples per SNR Level")
axes[1].set_xlabel("SNR (dB)")
axes[1].set_ylabel("Count")
axes[1].set_xticks(SNR_LEVELS)
axes[1].tick_params(axis='x', labelsize=7, rotation=45)

# (c) Signal magnitude example (I and Q over time)
sample_iq = Xs[500]   # pick sample 500
time_axis = np.arange(sample_iq.shape[1])
axes[2].plot(time_axis[:50], sample_iq[0, :50],
             label='I (In-phase)',   color='blue',   linewidth=1)
axes[2].plot(time_axis[:50], sample_iq[1, :50],
             label='Q (Quadrature)', color='orange', linewidth=1)
axes[2].set_title(f"I/Q Signal over Time\n"
                  f"({CLASS_NAMES[int(labels[500])]} "
                  f"@ SNR={int(snrs[500])}dB) — first 50 symbols")
axes[2].set_xlabel("Symbol Index")
axes[2].set_ylabel("Amplitude")
axes[2].legend(fontsize=8)
axes[2].grid(True, linewidth=0.4)

plt.tight_layout()
plt.savefig("plot3_statistics.png", dpi=150)
plt.show()
print(" Saved → plot3_statistics.png")

print("\n All plots generated successfully!")