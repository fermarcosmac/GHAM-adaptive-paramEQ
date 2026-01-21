import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
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
