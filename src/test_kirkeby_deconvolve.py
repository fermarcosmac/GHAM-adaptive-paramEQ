import math
from pathlib import Path

import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt

from experiment_03 import kirkeby_deconvolve


def generate_test_system(sr: int, length_ir: int = 1024, decay_time: float = 0.05) -> torch.Tensor:
    """Generate a simple synthetic LTI system impulse response.

    Uses an exponentially decaying noise IR, optionally windowed, to roughly
    resemble a small-room RIR. Output shape: (N,). """
    device = torch.device("cpu")
    t = torch.arange(length_ir, device=device) / sr
    decay = torch.exp(-t / decay_time)
    noise = torch.randn(length_ir, device=device)
    ir = decay * noise

    # Optional Hann window to taper both ends
    window = torch.hann_window(length_ir, periodic=False, device=device)
    ir = ir * window

    # Normalize to unit peak
    ir = ir / (ir.abs().max() + 1e-12)
    return ir


def generate_excitation(sr: int, duration_s: float = 1.0) -> torch.Tensor:
    """Generate an arbitrary excitation signal (white noise)."""
    device = torch.device("cpu")
    num_samples = int(duration_s * sr)
    x = torch.randn(num_samples, device=device)
    # Normalize to unit RMS
    x = x / (x.pow(2).mean().sqrt() + 1e-12)
    return x


def run_kirkeby_test(
    sr: int = 48000,
    roi: tuple = (100.0, 12000.0),
    ir_len: int = 1024,
    exc_duration_s: float = 0.01,
):
    """Run a basic performance test of kirkeby_deconvolve.

    - Synthesizes a random decaying IR h_true
    - Generates a white-noise excitation x
    - Forms y = x * h_true via FFT convolution
    - Applies kirkeby_deconvolve(x, y, nfft, sr, roi)
    - Compares estimated H and h_est to the ground truth in magnitude and time
    """
    torch.manual_seed(123)

    # Generate system and signals
    h_true = generate_test_system(sr, length_ir=ir_len)
    x = generate_excitation(sr, duration_s=exc_duration_s)

    # Convolution to get output y
    # Use full-length FFT conv via torchaudio for consistency with main script
    x_4d = x.view(1, 1, -1)
    h_4d = h_true.view(1, 1, -1)
    y_4d = torchaudio.functional.fftconvolve(x_4d, h_4d, mode="full")
    y = y_4d.view(-1)

    # Choose nfft similar to MATLAB example: ~2 * len(y), rounded so it isn't too small
    nfft = 2 * y.numel()

    # Run Kirkeby deconvolution
    H_est = kirkeby_deconvolve(x, y, nfft, sr, roi)

    # Ground-truth frequency response
    H_true = torch.fft.rfft(h_true, n=nfft)

    # Compute magnitude error over ROI
    freqs = torch.fft.rfftfreq(nfft, d=1.0 / sr)
    roi_mask = (freqs >= roi[0]) & (freqs <= roi[1])

    eps_mag = 1e-12
    H_true_mag_db = 20.0 * torch.log10(H_true.abs() + eps_mag)
    H_est_mag_db = 20.0 * torch.log10(H_est.abs() + eps_mag)

    err_mag_db = (H_est_mag_db - H_true_mag_db)[roi_mask]
    mae_mag_db = err_mag_db.abs().mean().item()
    rmse_mag_db = err_mag_db.pow(2).mean().sqrt().item()

    # Time-domain estimate via irfft and cropping to IR length
    h_est_full = torch.fft.irfft(H_est, n=nfft)
    h_est = h_est_full[: ir_len]

    # Align scales for fair comparison
    # Scale h_est so that its energy in ROI matches h_true approximately
    scale = (h_true @ h_est) / (h_est @ h_est + 1e-12)
    h_est_scaled = h_est * scale

    td_l2 = (h_est_scaled - h_true).pow(2).sum().sqrt().item()
    td_l2_norm = td_l2 / (h_true.pow(2).sum().sqrt().item() + 1e-12)

    print("Kirkeby deconvolution test")
    print(f"  Sample rate       : {sr} Hz")
    print(f"  ROI               : {roi[0]:.1f} - {roi[1]:.1f} Hz")
    print(f"  IR length         : {ir_len} samples")
    print(f"  Excitation length : {x.numel()} samples")
    print(f"  nfft              : {nfft}")
    print("  Magnitude error in ROI (dB):")
    print(f"    MAE  : {mae_mag_db:.3f} dB")
    print(f"    RMSE : {rmse_mag_db:.3f} dB")
    print("  Time-domain IR error (scaled estimate):")
    print(f"    L2 norm       : {td_l2:.3e}")
    print(f"    Relative L2   : {td_l2_norm:.3e}")

    # Plot frequency responses and impulse responses for visual inspection
    freqs_np = freqs.detach().cpu().numpy()
    H_true_mag_db_np = H_true_mag_db.detach().cpu().numpy()
    H_est_mag_db_np = H_est_mag_db.detach().cpu().numpy()

    h_true_np = h_true.detach().cpu().numpy()
    h_est_np = h_est_scaled.detach().cpu().numpy()
    t_ir = np.arange(ir_len) / sr

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Frequency-domain plot
    ax1.semilogx(freqs_np, H_true_mag_db_np, label="True H(f)", linewidth=1.0)
    ax1.semilogx(freqs_np, H_est_mag_db_np, label="Estimated H(f)", linewidth=1.0, alpha=0.8)
    ax1.set_xlim([20, sr / 2])
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude (dB)")
    ax1.set_title("Kirkeby Deconvolution: Magnitude Response")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    # Time-domain plot
    ax2.plot(t_ir, h_true_np, label="True IR", linewidth=1.0)
    ax2.plot(t_ir, h_est_np, label="Estimated IR (scaled)", linewidth=1.0, alpha=0.8)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Kirkeby Deconvolution: Impulse Responses")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_kirkeby_test()
