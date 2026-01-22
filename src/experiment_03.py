import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch, torchaudio
import torch.nn.functional as F
from scipy.signal import firls, freqz
from tqdm import tqdm
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from lib.local_dasp_pytorch.modules import ParametricEQ
from utils import (
    load_audio,
    load_rirs,
    ensure_rirs_sample_rate,
    get_delay_from_ir,
    next_power_of_2,
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
    frame_len = 2048*4  # Length (samples) of processing buffers
    hop_len = frame_len  # Stride between frames
    mu_cont = 0.001  # Learning rate for controller (normalized later)

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
    LEM = torch.from_numpy(rir).view(1,1,-1)
    LEM_memory = LEM.shape[-1]

    # Initialize differentiable EQ
    EQ = ParametricEQ(sample_rate=sr)
    #init_params_tensor = torch.rand(1,EQ.num_params) # random initialization
    # Uncomment lines below to use initial compenation EQ params as initialization
    dasp_param_dict = { k: torch.as_tensor(v, dtype=torch.float32).view(1) for k, v in EQ_comp_dict["eq_params"].items() }
    _, init_params_tensor = EQ.clip_normalize_param_dict(dasp_param_dict) # initial normalized parameter vector
    EQ_params = torch.nn.Parameter(init_params_tensor.clone())
    EQ_memory = 3 # TODO: hardcoded for now

    # Load/synthesise the input audio (as torch tensors)
    if input_type == "white_noise":
        # Generate white noise excitation
        T = int(max_audio_len_s * sr)
        torch.manual_seed(123)  # Seed for reproducibility
        input = torch.randn(T)
        print(f"Input signal: White noise ({max_audio_len_s} s, {T} samples)")
    else:
        # Load audio file from data/audio/input/
        audio_path = audio_input_dir / input_type
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        input, audio_sr = torchaudio.load(audio_path)
        
        # Convert to mono if stereo (average channels)
        if input.shape[0] > 1:
            input = input.mean(dim=0)
        else:
            input = input.squeeze(0)
        
        # Resample if necessary to match RIR sample rate
        if audio_sr != sr:
            resampler = torchaudio.transforms.Resample(orig_freq=audio_sr, new_freq=sr)
            input = resampler(input)
            print(f"Resampled audio from {audio_sr} Hz to {sr} Hz")
        
        # Truncate to max_audio_len_s if specified
        if max_audio_len_s is not None:
            max_samples = int(max_audio_len_s * sr)
            if len(input) > max_samples:
                input = input[:max_samples]
                print(f"Truncated audio to first {max_audio_len_s} seconds")
        
        T = len(input)
        T_seconds = T / sr
        print(f"Input signal: {input_type} ({T_seconds:.2f} s, {T} samples)")

    # Normalize input signal and adapt tensor shape
    input = input / input.abs().max()
    input = input.view(1,1,-1)

    # Allocate results buffers
    y_control = torch.zeros(1,1,T)

    # Initialize buffers
    in_buffer = torch.zeros(1,1,frame_len)
    EQ_out_len = next_power_of_2(frame_len + EQ_memory - 1)
    EQ_out_buffer = torch.zeros(1,1,EQ_out_len)
    LEM_out_len = frame_len + LEM_memory - 1
    LEM_out_buffer = torch.zeros(1,1,LEM_out_len)

    # Hanning window for overlap-add (use 50% overlap for perfect reconstruction)
    window = torch.hann_window(frame_len, periodic=True).view(1, 1, -1)

    mu = mu_cont / frame_len # TODO: check step size normalization carefully!

    # Main loop
    n_frames = (T - frame_len) // hop_len + 1
    for k in tqdm(range(n_frames), desc="Processing", unit="frame"):

        start_idx = k * hop_len

        # Update input buffer and apply window
        window = 1
        in_buffer = input[:,:,start_idx:start_idx+frame_len] * window

        # Process through EQ
        EQ_out = EQ.process_normalized(in_buffer, EQ_params)

        # Update EQ output buffer (shift left by hop_len and add new samples)
        EQ_out_buffer = F.pad(EQ_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
        EQ_out_buffer += EQ_out
        #EQ_out_buffer[..., :frame_len] += in_buffer # DEBUG (don't EQ at all)

        # Process through LEM
        #LEM_out = F.conv1d(EQ_out_buffer, LEM, padding=LEM_memory-1)
        LEM_out = torchaudio.functional.fftconvolve(EQ_out_buffer[:,:,:frame_len], LEM.view(1,1,-1), mode="full")
        
        # Update LEM output buffer (shift left by hop_len)
        LEM_out_buffer = F.pad(LEM_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
        LEM_out_buffer += LEM_out

        # Use LEM output to compute loss and update EQ parameters
        # TODO

        # Store output frame (only store hop_len new samples to handle overlap-add)
        end_idx = min(start_idx + frame_len, T)
        samples_to_store = end_idx - start_idx
        y_control[:, :, start_idx:end_idx] += LEM_out_buffer[:, :, :samples_to_store]




    #%% SAVE AUDIO FILES
    
    # Create output directory
    output_dir = base / "data" / "audio" / "ex_03"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare signals for saving (squeeze to 1D and normalize)
    input_save = input.squeeze()
    input_save = input_save / input_save.abs().max() * 0.9
    
    y_control_save = y_control.squeeze()
    y_control_save = y_control_save / (y_control_save.abs().max() + 1e-10) * 0.9
    
    # Save audio files
    torchaudio.save(output_dir / "input.wav", input_save.unsqueeze(0).cpu(), sr)
    torchaudio.save(output_dir / "output_controlled.wav", y_control_save.unsqueeze(0).cpu().detach(), sr)
    
    print(f"\nAudio files saved to {output_dir}:")
    print(f"  - input.wav: Input signal")
    print(f"  - output_controlled.wav: Output after EQ and LEM")

    #plt.show()