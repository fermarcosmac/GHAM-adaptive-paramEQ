"""
adaptive_equalization_skeleton.py

Skeleton code to simulate a time-varying room impulse response (RIR)
applied to an input audio file using an overlap-add scheme.

Behavior:
- Load an audio file from data/audio
- Load RIRs from data/rir (we'll pick two by default)
- The user supplies a sequence of RIR indices and corresponding start times
  (in seconds) describing when each RIR becomes active.
- The audio is processed in short analysis frames (window/hop). For each
  frame we pick the active RIR (by frame midpoint) and convolve the frame
  with that RIR. The convolved chunks are overlap-added into the output
  buffer.

This is a starting skeleton; next steps could add crossfading, adaptive
filtering/estimation, batch FFT convolution optimizations, GPU acceleration,
or variable window sizes.

Dependencies:
- numpy
- scipy
- soundfile (pysoundfile)

Install (if needed):
    pip install numpy scipy soundfile

"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly
import bisect
import warnings
import matplotlib.pyplot as plt


def load_audio(path: Path) -> Tuple[np.ndarray, int]:
    """Load audio as float32 mono.

    Returns:
        audio (np.ndarray, shape=(N,)): floating point audio in range [-1, 1]
        sr (int): sample rate
    """
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        # Convert to mono by averaging channels (simple default)
        data = data.mean(axis=1)
    # ensure float32
    data = data.astype(np.float32)
    return data, sr


def load_rirs(rir_dir: Path, max_n: int = None) -> Tuple[List[np.ndarray], List[int]]:
    """Load all wav files in a directory as float32 RIRs. Returns (rirs, srs).

    RIRs are returned as lists of 1-D numpy arrays. Sample rates are returned
    for each file so the caller can check/resample if needed.
    """
    files = sorted([p for p in rir_dir.glob("*.wav")])
    if max_n is not None:
        files = files[:max_n]
    rirs = []
    srs = []
    for f in files:
        data, sr = sf.read(str(f))
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)
        rirs.append(data)
        srs.append(sr)
    return rirs, srs


def ensure_rirs_sample_rate(rirs: List[np.ndarray], rirs_srs: List[int], target_sr: int) -> List[np.ndarray]:
    """Resample RIRs to match the target sample-rate if needed.

    Uses a simple resample_poly approach (from scipy.signal) for reasonable quality.
    """
    out = []
    for rir, sr in zip(rirs, rirs_srs):
        if sr == target_sr:
            out.append(rir)
        else:
            # Compute integer factors for resample_poly if possible
            # fallback to float ratio using resample_poly with nearest ints
            gcd = np.gcd(sr, target_sr)
            up = target_sr // gcd
            down = sr // gcd
            rir_rs = resample_poly(rir, up, down).astype(np.float32)
            warnings.warn(f"Resampled RIR from {sr} Hz -> {target_sr} Hz; length {len(rir)} -> {len(rir_rs)}")
            out.append(rir_rs)
    return out


def _active_rir_index_for_time(start_times_s: List[float], t: float) -> int:
    """Given a sorted list of start times (seconds) and a time t (seconds),
    return the index of the active RIR. The active RIR is the last index i
    such that start_times_s[i] <= t. If t < start_times_s[0], returns 0.
    """
    # bisect_right returns insertion point; subtract 1 to get index <= t
    i = bisect.bisect_right(start_times_s, t) - 1
    if i < 0:
        return 0
    return i


def simulate_time_varying_rir(
    audio: np.ndarray,
    sr: int,
    rirs: List[np.ndarray],
    rir_indices: List[int],
    start_times_s: List[float],
    window_ms: float = 100.0,
    hop_ms: float = 50.0,
) -> Tuple[np.ndarray, int]:
    """Simulate a time-varying RIR by processing the input audio in frames and
    convolving each frame with the active RIR. Overlap-add is used to reconstruct
    the final signal.

    Args:
        audio: 1-D numpy array (float32)
        sr: sample rate
        rirs: list of RIR arrays (already resampled to sr)
        rir_indices: list of indices into `rirs` indicating the sequence of RIRs
        start_times_s: list of start times in seconds for each corresponding index.
                       Must be same length as rir_indices, sorted ascending, and
                       first element normally 0.0 (but not required).
        window_ms: analysis window length in milliseconds
        hop_ms: hop size in milliseconds

    Returns:
        y: processed audio (float32)
        sr: sample rate
    """
    assert len(rir_indices) == len(start_times_s), "rir_indices and start_times_s must have same length"
    # sort/check consistency
    # create a mapping of absolute start_times for the sequence
    seq_start_times = list(start_times_s)
    if any(t < 0 for t in seq_start_times):
        raise ValueError("start_times_s must be non-negative")
    if any(seq_start_times[i] > seq_start_times[i + 1] for i in range(len(seq_start_times) - 1)):
        raise ValueError("start_times_s must be sorted ascending")

    # Convert to sample-based window/hop
    win_len = int(round(window_ms * sr / 1000.0))
    hop_len = int(round(hop_ms * sr / 1000.0))
    if win_len <= 0 or hop_len <= 0:
        raise ValueError("window_ms and hop_ms must be positive and produce non-zero lengths")

    n = len(audio)
    max_rir_len = max(len(r) for r in rirs)
    conv_len = win_len + max_rir_len - 1
    # Output length must accommodate convolution tail
    out_len = n + win_len +max_rir_len
    y = np.zeros(out_len, dtype=np.float32)

    # Frame starts
    frame_starts = list(range(0, n, hop_len))

    for s in frame_starts:              # start index
        e = s + win_len                 # end index  
        frame = audio[s:e]
        # If last frame is shorter, pad with zeros
        if len(frame) < win_len:
            frame = np.pad(frame, (0, win_len - len(frame)))

        # choose active RIR based on the midpoint of the frame
        midpoint_s = (s + win_len // 2) / sr        # I THINK I CAN DO IT WITH START INDEX!
        # Determine which index in rir_indices is active
        seq_idx = _active_rir_index_for_time(seq_start_times, midpoint_s)
        rir_idx = rir_indices[seq_idx]      # I THINK THIS IS INNEFICIENT
        rir = rirs[rir_idx]

        # convolve frame with the selected RIR
        conv = fftconvolve(frame, rir, mode="full").astype(np.float32)

        # add to output at the correct place
        try:
            y[s:s+conv_len] += conv
        except:
            hey=0

    # Trim the output to reasonable length (you might prefer to keep the full tail)
    # Here we trim to original audio length + max_rir_len
    y = y[: n + max_rir_len]

    return y, sr

def rms(x: np.ndarray) -> float:
    """Compute root-mean-square of a 1-D numpy array."""
    return np.sqrt(np.mean(x**2))


def save_audio(path: Path, data: np.ndarray, sr: int) -> None:
    """Save audio as 32-bit float WAV (or default by soundfile)."""
    sf.write(str(path), data, sr)


if __name__ == "__main__":
    # Set paths
    base = Path(".")
    audio_path = base / "data" / "audio" / "input" / "guitar-riff.wav"
    rir_dir = base / "data" / "rir"
    out_path = base / "data" / "audio" / "output" / "output.wav"

    # Load audio
    audio, sr = load_audio(audio_path)
    duration_s = len(audio) / sr
    print(f"Loaded audio: {audio_path}, {len(audio)} samples, {sr} Hz, {duration_s:.2f} s")

    # Load RIRs (pick 2 for now)
    rirs, rirs_srs = load_rirs(rir_dir, max_n=2)
    if len(rirs) < 2:
        raise RuntimeError("Need at least two RIR files in data/rir for this demo")

    # Ensure RIRs have same sample rate as audio
    rirs = ensure_rirs_sample_rate(rirs, rirs_srs, sr)

    # Define RIR sequence and start times (seconds)
    # Example: use RIR 0 from 0s, switch to RIR 1 at halfway through the audio
    rir_indices = [0, 1]
    start_times_s = [0.0, duration_s / 2.0]

    # Simulate
    y, _ = simulate_time_varying_rir(audio, sr, rirs, rir_indices, start_times_s, window_ms=100, hop_ms=50)
    y = y*rms(audio)/rms(y)  # normalize output to input RMS

    # Save result
    #save_audio(out_path, y, sr)
    #print(f"Saved simulated time-varying output to: {out_path}")

    # Plot results
    time_axis = np.arange(0, len(y)) / sr
    plt.figure()
    plt.plot(time_axis[:len(audio)],audio, label="Input")
    plt.plot(time_axis, y, label="Variable Room Output")
    # draw vertical change-point lines, add label only once
    for i, xt in enumerate(start_times_s):
        if i == 0:
            pass # First RIR is not a "change"
        else:
            plt.axvline(xt, color="red", linestyle="--")
    plt.title("Input and Time-Varying RIR Output")
    plt.legend()
    plt.show()