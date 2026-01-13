import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
# Ensure the workspace root is first on sys.path so the local package is imported
root = Path(__file__).resolve().parent.parent  # src -> repo root
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from local_dasp_pytorch.modules import ParametricEQ
from modules import EQController_dasp, EQLogger
from utils import (
    load_audio,
    get_compensation_EQ_params,
    load_rirs,
    ensure_rirs_sample_rate,
    simulate_time_varying_process,
)



# Define process function (that defines the nature of the experiment) here!


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
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs)
    rirs = ensure_rirs_sample_rate(rirs, rirs_srs, sr)

    # Parametric EQ estimation for virtual room compensation
    rir_init = rirs[0]
    EQ_comp_dict = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=6)

    # Prepare differentiable EQ module with initial compensation parameters
    EQ = ParametricEQ(sample_rate=sr)
    dasp_param_dict = { k: torch.as_tensor(v, dtype=torch.float32).view(1) for k, v in EQ_comp_dict["eq_params"].items() }
    _, init_params_tensor = EQ.clip_normalize_param_dict(dasp_param_dict) # initial normalized parameter vector

    # Prepare controller and logger for adaptive EQ (to be defined)
    # TODO
    EQController_dasp = EQController_dasp() # Will take EQ module with its initial parameters and implement adaptation logic
    EQLogger = EQLogger()                   # Maybe it should be an attribute of the EQController_dasp class...

    # I also have to define the process_fn, which should:
    #     - take in input frame, sr, rir, frame_start, frame_idx
    #     - apply EQ to input frame
    #     - convolve EQed frame with rir (step 3)
    #     - update EQ parameters calling controller and updating its state
    #     - call logger to log parameters and performance
    #     - return processed frame (the output of step 3)

    # Playback simulation:
    switch_times_s = [t * (len(input) / sr) for t in switch_times_norm]
    rir_indices = list(range(n_rirs))  # use RIRS in order
    y, sr = simulate_time_varying_process(
        audio=input,
        sr=sr,
        rirs=rirs,
        rir_indices=rir_indices,
        start_times_s=switch_times_s,
        EQ=EQ,
        controller=EQController_dasp,
        logger=EQLogger,)

    # plot and show input, output signals and switch times
    time_axis = np.arange(0, len(y)) / sr
    plt.figure()
    plt.plot(time_axis[:len(input)],input, label="Input")
    plt.plot(time_axis, y.detach().numpy(), label="Variable Room Output")
    for i, xt in enumerate(switch_times_s):
        plt.axvline(x=xt, color='k', linestyle='--', label="RIR Switch" if i==0 else None)
    plt.title("Input and Variable Room Output Signals")
    plt.legend()
    plt.show()

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