import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch, torchaudio
from scipy.signal import firls, freqz
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from utils import (
    load_audio,
    load_rirs,
    ensure_rirs_sample_rate,
    get_delay_from_ir,
    save_audio,
    get_compensation_EQ_params,
)

# NOTE 1: careful with the gradients you compute. If different samples are generated using different EQ parameters

#%% MAIN SCRIPT

if __name__ == "__main__":

    # Set paths
    base = Path(".")
    rir_dir = base / "data" / "rir"
    audio_input_dir = base / "data" / "audio" / "input"

    # Input configuration
    input_type = "onde_day_funk.wav" # Either a file or a valid synthesisable signal
    max_audio_len_s = 10.0  # None = full length

    # Simulation configuration
    ROI = [100.0, 14000.0]  # region of interest for EQ compensation (Hz)
    frame_len = 8192 # Length (samples) of processing buffers

    # Acoustic path from actuator (speaker) to sensor (microphone)
    rirs, rirs_srs = load_rirs(rir_dir, max_n=1)
    rir = rirs[0]
    sr = rirs_srs[0]

    # Desired response computation: delay and magnitude response
    lem_delay = get_delay_from_ir(rir, sr)
    EQ_comp_dict = get_compensation_EQ_params(rir, sr, ROI, num_sections=6)
    target_mag_resp = EQ_comp_dict["target_response_db"]
    target_mag_freqs = EQ_comp_dict["freq_axis_smoothed"]
    # TODO: interpolate to actual FFT bin used later (frame_len)


    # For now, we assume that the LEM is well identified, so we just use the rir as is
    # Initialize the LEM estimate
    LEM = rir
    LEM_memory = len(LEM)

    # Initialize differentiable EQ
    EQ_memory = 0.0 # TODO: hardcoded for now

    # Load/synthesise the input audio
    if input_type == "white_noise":
        # Generate white noise excitation
        T = int(max_audio_len_s * sr)
        np.random.seed(123)  # Seed for reproducibility
        input = np.random.randn(T)
        print(f"Input signal: White noise ({max_audio_len_s} s, {T} samples)")
    else:
        # Load audio file from data/audio/input/
        audio_path = audio_input_dir / input_type
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        input, audio_sr = load_audio(audio_path)
        
        # Resample if necessary to match RIR sample rate
        if audio_sr != sr:
            from scipy.signal import resample
            new_len = int(len(input) * sr / audio_sr)
            input = resample(input, new_len)
            print(f"Resampled audio from {audio_sr} Hz to {sr} Hz")
        
        # Convert to mono if stereo
        if input.ndim > 1:
            input = np.mean(input, axis=1)
        
        # Truncate to max_audio_len_s if specified
        if max_audio_len_s is not None:
            max_samples = int(max_audio_len_s * sr)
            if len(input) > max_samples:
                input = input[:max_samples]
                print(f"Truncated audio to first {max_audio_len_s} seconds")
        
        T = len(input)
        T_seconds = T / sr
        print(f"Input signal: {input_type} ({T_seconds:.2f} s, {T} samples)")

    # Normalize input signal
    input = input / np.max(np.abs(input))

    # Allocate results buffers
    y_control = torch.zeros(T)
    
    # Initialize buffers
    in_buffer = torch.zeros(frame_len)
    EQ_out_buffer = torch.zeros(frame_len+EQ_memory-1)
    LEM_out_buffer = torch.zeros(EQ_out_buffer+LEM_memory-1)