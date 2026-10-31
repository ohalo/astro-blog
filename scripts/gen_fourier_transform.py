#!/usr/bin/env python3
"""Generate 7 real computation images for fourier-transform-volatility-signal article."""
import numpy as np
import json
import os

np.random.seed(20261031)
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'public', 'images', 'fourier-transform-volatility-signal')
os.makedirs(output_dir, exist_ok=True)

# ============ Generate data ============
T = 1000
dt = 1.0
t = np.arange(T)

theta_clean = (0.015 + 0.005 * np.cos(2 * np.pi * t / 60)
               + 0.003 * np.cos(2 * np.pi * t / 20))
noise = np.random.normal(0, 0.003, T)
theta_obs = theta_clean + noise

# FFT
freqs_full = np.fft.fftfreq(T, dt)
fft_coeffs_full = np.fft.fft(theta_obs)
fft_vals_full = np.abs(fft_coeffs_full) ** 2 / T

# 正频率
pos_full = freqs_full > 0
freqs_pos = freqs_full[pos_full]
psd = fft_vals_full[pos_full]

top3_idx_pos = np.argsort(psd)[-3:][::-1]
top3_idx_full = np.where(pos_full)[0][top3_idx_pos]

# Bandpass filter: keep signal band (0.008-0.06) and DC (freq=0)
f_low = 0.008
f_high = 0.06
filtered_fft_coeffs = np.copy(fft_coeffs_full)
# Keep DC (index 0)
# Zero out negative frequencies and high/low positive frequencies
filtered_fft_coeffs[~pos_full & (freqs_full != 0)] = 0  # keep DC, remove neg freqs
filtered_fft_coeffs[pos_full & (freqs_full < f_low)] = 0
filtered_fft_coeffs[pos_full & (freqs_full > f_high)] = 0
theta_filtered = np.real(np.fft.ifft(filtered_fft_coeffs))

mse_raw = float(np.mean((theta_obs - theta_clean) ** 2))
mse_filtered = float(np.mean((theta_filtered - theta_clean) ** 2))

# STFT data
T2 = 1000
t2 = np.arange(T2)
theta_clean2 = np.zeros(T2)
theta_clean2[:500] = 0.015 + 0.005 * np.cos(2 * np.pi * t2[:500] / 60)
theta_clean2[500:] = 0.015 + 0.003 * np.cos(2 * np.pi * (t2[500:] - 500) / 20)
noise2 = np.random.normal(0, 0.002, T2)
theta_obs2 = theta_clean2 + noise2

window_length = 200
window = np.hanning(window_length)
overlap = int(window_length * 0.5)
n_steps = (T2 - window_length) // overlap + 1

freqs_stft = np.fft.fftfreq(window_length, dt)
pos_stft = freqs_stft > 0
freqs_pos_stft = freqs_stft[pos_stft]
pos_idx_stft = np.where(pos_stft)[0]

stft_matrix = np.zeros((len(freqs_pos_stft), n_steps))
for i in range(n_steps):
    start = i * overlap
    segment = theta_obs2[start:start + window_length] * window
    spectrum = np.abs(np.fft.fft(segment)) ** 2 / window_length
    stft_matrix[:, i] = spectrum[pos_idx_stft]

periods_stft = 1.0 / freqs_pos_stft
top_freq_idx = np.argmax(stft_matrix, axis=0)
dominant_periods = periods_stft[top_freq_idx]

results = {
    'mse_raw': round(mse_raw, 6),
    'mse_filtered': round(mse_filtered, 6),
    'improvement_pct': round(100 * (1 - mse_filtered / mse_raw), 1),
    'T': T,
    'periods_detected': [round(1.0 / freqs_pos[idx_pos], 1) for idx_pos in top3_idx_pos]
}

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============ Fig 1: Power spectral density ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freqs_pos, psd, 'b-', lw=1.2, alpha=0.8)
for i, idx_pos in enumerate(top3_idx_pos):
    lbl = f'{periods_stft[idx_pos]:.0f} day' if i == 0 else None
    ax.axvline(x=freqs_pos[idx_pos], color='red', ls='--', alpha=0.5, label=lbl)
ax.set_xlabel('Frequency (1/days)')
ax.set_ylabel('Power Spectral Density')
ax.set_title('Power Spectral Density of Volatility Signal')
ax.set_xlim(0, 0.1)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig01.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 2: Time domain vs filtered ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(T), theta_clean, 'b-', lw=1.5, label='True volatility', alpha=0.8)
ax.plot(range(T), theta_obs, 'g-', lw=0.8, label='Observed (noisy)', alpha=0.6)
ax.plot(range(T), theta_filtered, 'r-', lw=1.2, label='Bandpass filtered', alpha=0.9)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Volatility')
ax.set_title('FFT Bandpass Filtering: Removing High-Frequency Noise')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig02.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 3: FFT complex magnitude spectrum ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(freqs_pos, np.sqrt(psd), 'b-', lw=1.2)
ax.set_xlabel('Frequency (1/days)')
ax.set_ylabel('Magnitude |X(f)|')
ax.set_title('FFT Complex Magnitude Spectrum')
ax.set_xlim(0, 0.1)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig03.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 4: Different cutoff frequencies ============
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
for idx, (cutoff_val, title) in enumerate([(0.05, '5%'), (0.1, '10%'), (0.15, '15%'), (0.2, '20%')]):
    ax_row = idx // 2
    ax_col = idx % 2
    cutoff_f = freqs_pos[int(cutoff_val * len(freqs_pos))]
    filt_fft = np.copy(fft_coeffs_full)
    filt_fft[0] = fft_coeffs_full[0]  # keep DC
    filt_fft[~pos_full] = 0
    filt_fft[pos_full & (freqs_full > cutoff_f)] = 0
    filt_signal = np.real(np.fft.ifft(filt_fft))
    ax[ax_row, ax_col].plot(range(T), theta_clean, 'b-', lw=1.2, alpha=0.6, label='True')
    ax[ax_row, ax_col].plot(range(T), filt_signal, 'r-', lw=1, label=f'{title} cutoff')
    ax[ax_row, ax_col].set_title(f'{title} Cutoff')
    ax[ax_row, ax_col].set_xlabel('Time')
    ax[ax_row, ax_col].legend(fontsize=8)
    ax[ax_row, ax_col].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'fig04.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 5: Period component decomposition ============
fig, ax = plt.subplots(figsize=(10, 5))
filt_60 = np.copy(fft_vals_full)
filt_60[0] = fft_vals_full[0]
filt_60[~pos_full] = 0
filt_60[pos_full & (np.abs(freqs_full - 1/60) >= 0.003)] = 0
comp_60 = np.real(np.fft.ifft(filt_60))

filt_20 = np.copy(fft_vals_full)
filt_20[0] = fft_vals_full[0]
filt_20[~pos_full] = 0
filt_20[pos_full & (np.abs(freqs_full - 1/20) >= 0.003)] = 0
comp_20 = np.real(np.fft.ifft(filt_20))

ax.plot(range(T), comp_60, 'b-', lw=1.2, label='60-day component', alpha=0.8)
ax.plot(range(T), comp_20, 'r-', lw=1.2, label='20-day component', alpha=0.8)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Amplitude')
ax.set_title('Periodic Component Decomposition')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig05.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 6: Time-frequency waterfall ============
fig, ax = plt.subplots(figsize=(10, 6))
freqs_display = freqs_pos_stft * 1000
time_display = np.arange(n_steps) * overlap
mesh = ax.pcolormesh(time_display, freqs_display, stft_matrix, shading='gouraud', cmap='viridis')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Frequency (1/day)')
ax.set_title('Short-Time Fourier Transform (Time-Frequency)')
fig.colorbar(mesh, label='Power')
ax.grid(alpha=0.3)
fig.savefig(os.path.join(output_dir, 'fig06.png'), dpi=150, bbox_inches='tight')
plt.close()

# ============ Fig 7: Time-varying period tracking ============
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(0, n_steps * overlap, overlap), dominant_periods, 'r-', lw=1.5, label='Dominant period (STFT)')
ax.axhline(y=60, color='b', ls='--', alpha=0.5, label='60-day reference')
ax.axhline(y=20, color='g', ls='--', alpha=0.5, label='20-day reference')
ax.axvline(x=500, color='gray', ls=':', alpha=0.5)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Dominant Period (days)')
ax.set_title('Time-Varying Period Tracking via STFT')
ax.legend(loc='upper right', fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(10, 80)
fig.savefig(os.path.join(output_dir, 'fig07.png'), dpi=150, bbox_inches='tight')
plt.close()

# Save results JSON
json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         '_ft_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Generated 7 images in {output_dir}")
print(f"Results: {json.dumps(results, indent=2)}")
