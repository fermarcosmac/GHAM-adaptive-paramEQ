from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy as np
from utils import (
    load_audio,
    save_audio,
    get_compensation_EQ_params,
    load_rirs,
    ensure_rirs_sample_rate,
    simulate_time_varying_rir,
    rms,
)






# Take all the code above to utils.py! TODO




#%% MAIN SCRIPT

if __name__ == "__main__":

    # Set paths
    base = Path(".")
    audio_path = base / "data" / "audio" / "input" / "guitar-riff.wav"
    rir_dir = base / "data" / "rir"

    # Set experiment parameters
    n_rirs = 2  # number of RIRs to use
    switch_times_norm = [0.0, 0.5]  # times to switch RIRs (normalized)
    ROI = [150.0, 14000.0]  # region of interest for EQ compensation (Hz)

    # Load probe and ground-truth RIR
    input, sr = load_audio(audio_path) # input audio signal
    rirs, rirs_srs = load_rirs(rir_dir, max_n=1)
    rirs = ensure_rirs_sample_rate(rirs, rirs_srs, sr)

    # Parametric EQ estimation for virtual room compensation
    rir_init = rirs[0]
    EQ_comp_dict = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=6)

    

    # Playback simulation:
    #    - Static room - no EQ
    #    - Static room + initial EQ
    #    - Dynamic room - no EQ
    #    - Dynamic room + adaptive EQ

    # Common interface for different adaptive EQ settings (my class, torch.optim?)
    #    - Pytorch optimizers (including Newton library)
    #    - GHAM adaptive filter
    #    - Neural network-based adaptive EQ (MLP/RNN)

    # They should take in the error signal and output EQ parameters at each correction step
    # Make sure each correctioin step fits in real-time constraints (e.g., 10 ms processing time budget)

    pass