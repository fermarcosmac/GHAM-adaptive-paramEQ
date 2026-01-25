import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch, torchaudio
import torch.nn.functional as F
from scipy.signal import firls, freqz, minimum_phase
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


def frame_analysis_plot(in_buffer, EQ_out, LEM_out, target_frame, frame_idx=None):
    """Plot 4 signals in a 4x1 subplot grid for frame-by-frame analysis.
    
    Args:
        in_buffer: Input buffer tensor (1, 1, frame_len)
        EQ_out: EQ output buffer tensor (1, 1, frame_len)
        LEM_out: LEM output buffer tensor (1, 1, frame_len)
        target_frame: Target frame tensor (1, 1, frame_len)
        frame_idx: Optional frame index for title
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    
    title = f"Frame Analysis (Frame {frame_idx})" if frame_idx is not None else "Frame Analysis"
    fig.suptitle(title)
    
    # Convert tensors to numpy for plotting
    in_np = in_buffer.squeeze().cpu().detach().numpy()
    eq_np = EQ_out.squeeze().cpu().detach().numpy()
    lem_np = LEM_out.squeeze().cpu().detach().numpy()
    target_np = target_frame.squeeze().cpu().detach().numpy()
    
    axes[0].plot(in_np)
    axes[0].set_ylabel("Input Buffer")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(eq_np)
    axes[1].set_ylabel("EQ Output")
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(lem_np)
    axes[2].set_ylabel("LEM Output")
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(target_np)
    axes[3].set_ylabel("Target Frame")
    axes[3].set_xlabel("Sample")
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def build_desired_response_lin_phase(sr: int, response_type: str = "delay_only",
                           target_mag_resp: np.ndarray = None, target_mag_freqs: np.ndarray = None, 
                           fir_len: int = 1024, ROI: tuple = None, rolloff_octaves: float = 1.0,
                           device: torch.device = None):
    """Build a desired response FIR filter with optional target magnitude response.
    
    Args:
        sr: Sample rate in Hz
        response_type: Type of response - "delay_only" or "delay_and_mag"
        target_mag_resp: Target magnitude response in dB (1D array), required if response_type="delay_and_mag"
        target_mag_freqs: Frequency axis for target_mag_resp in Hz (1D array), required if response_type="delay_and_mag"
        fir_len: Length of the FIR filter for magnitude shaping (default 1024)
        ROI: Region of interest as (low_freq, high_freq) in Hz. Frequencies outside will be attenuated.
        rolloff_octaves: Width of the rolloff transition in octaves (default 1.0)
        device: Torch device for output tensor
        
    Returns:
        desired_response: Tensor of shape (1, 1, N) representing the desired impulse response
    """
    if device is None:
        device = torch.device("cpu")
    
    if response_type == "delay_only":
        # Simple Kronecker delta (no delay - add delay externally if needed)
        desired_response = torch.zeros(1, 1, 1, device=device)
        desired_response[:, :, 0] = 1.0
        print(f"Desired response: delay only (1 sample)")
        
    elif response_type == "delay_and_mag":
        if target_mag_resp is None or target_mag_freqs is None:
            raise ValueError("target_mag_resp and target_mag_freqs are required for response_type='delay_and_mag'")
        
        # Interpolate target magnitude response to FFT bins
        fft_freqs = np.fft.rfftfreq(fir_len, d=1.0/sr)
        
        # Interpolate target magnitude response (in dB) to FFT frequency bins
        target_mag_interp_db = np.interp(fft_freqs, target_mag_freqs, target_mag_resp)
        
        # Apply ROI rolloff: attenuate frequencies outside the region of interest
        # The target_mag_resp is kept as-is within ROI; decay starts exactly at ROI boundaries
        if ROI is not None:
            f_low, f_high = ROI
            rolloff_mask_db = np.zeros_like(fft_freqs)
            
            # Low-frequency decay (below f_low)
            # Decay starts at f_low (0 dB) and reaches -120 dB at f_low_end
            f_low_end = f_low / (2 ** rolloff_octaves)
            for i, f in enumerate(fft_freqs):
                if f <= 0:
                    rolloff_mask_db[i] = -120  # DC component heavily attenuated
                elif f < f_low_end:
                    rolloff_mask_db[i] = -120  # Full attenuation below rolloff end
                elif f < f_low:
                    # Smooth cosine transition from 0 dB (at f_low) to -120 dB (at f_low_end)
                    # t=0 at f_low, t=1 at f_low_end
                    t = np.log2(f_low / f) / rolloff_octaves  # 0 at f_low, 1 at f_low_end
                    rolloff_mask_db[i] = -120 * 0.5 * (1 - np.cos(np.pi * t))
            
            # High-frequency decay (above f_high)
            # Decay starts at f_high (0 dB) and reaches -120 dB at f_high_end
            f_high_end = f_high * (2 ** rolloff_octaves)
            for i, f in enumerate(fft_freqs):
                if f > f_high_end:
                    rolloff_mask_db[i] = -120  # Full attenuation above rolloff end
                elif f > f_high:
                    # Smooth cosine transition from 0 dB (at f_high) to -120 dB (at f_high_end)
                    # t=0 at f_high, t=1 at f_high_end
                    t = np.log2(f / f_high) / rolloff_octaves  # 0 at f_high, 1 at f_high_end
                    rolloff_mask_db[i] = -120 * 0.5 * (1 - np.cos(np.pi * t))
            
            # Apply rolloff mask to target magnitude
            target_mag_interp_db = target_mag_interp_db + rolloff_mask_db
            print(f"Applied ROI decay: {f_low:.0f}-{f_high:.0f} Hz, transition: {rolloff_octaves} octave(s)")
        
        # Convert from dB to linear magnitude
        target_mag_interp_linear = 10 ** (target_mag_interp_db / 20.0)
        
        # Create linear-phase FIR filter via IFFT (zero-phase, then shift for causality)
        H_mag = torch.from_numpy(target_mag_interp_linear).float().to(device)
        
        # Create zero-phase impulse response via IFFT
        h_zerophase = torch.fft.irfft(H_mag, n=fir_len)
        
        # Shift to make causal (linear phase) by circular shift
        h_causal = torch.roll(h_zerophase, fir_len // 2)
        
        # Apply a window to smooth the filter
        fir_window = torch.hann_window(fir_len, device=device)
        h_windowed = h_causal * fir_window
        
        # Normalize filter to unity gain at passband
        h_windowed = h_windowed / h_windowed.abs().max()
        
        desired_response = h_windowed.view(1, 1, -1)
        
        print(f"Desired response: {desired_response.shape[-1]} samples (FIR len={fir_len})")
    else:
        raise ValueError(f"Unknown response_type: {response_type}. Use 'delay_only' or 'delay_and_mag'")
    
    return desired_response


#%% MAIN SCRIPT

if __name__ == "__main__":

    # Set paths
    base = Path(".")
    rir_dir = base / "data" / "rir"
    audio_input_dir = base / "data" / "audio" / "input"

    # Input configuration
    input_type = "white_noise" # Either a file or a valid synthesisable signal
    max_audio_len_s = 10.0  # None = full length

    # Simulation configuration
    ROI = [100.0, 14000.0]  # region of interest for EQ compensation (Hz)
    frame_len = 1024  # Length (samples) of processing buffers
    hop_len = frame_len  # Stride between frames
    mu_cont = 0.001  # Learning rate for controller (normalized later)
    desired_response_type = "delay_and_mag"  # "delay_and_mag" or "delay_only"

    # Device selection (GPU if available)
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

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
    LEM = torch.from_numpy(rir).view(1,1,-1).to(device)
    LEM_memory = LEM.shape[-1]

    # Initialize differentiable EQ
    EQ = ParametricEQ(sample_rate=sr)
    torch.manual_seed(126)  # Seed for reproducibility
    init_params_tensor = torch.rand(1,EQ.num_params) # random initialization: It's pretty sensitive to initial parameters
    #init_params_tensor = torch.zeros(1,EQ.num_params) # random initialization
    # Uncomment lines below to use initial compenation EQ params as initialization
    dasp_param_dict = { k: torch.as_tensor(v, dtype=torch.float32).view(1) for k, v in EQ_comp_dict["eq_params"].items() }
    _, init_params_tensor = EQ.clip_normalize_param_dict(dasp_param_dict) # initial normalized parameter vector
    EQ_params = torch.nn.Parameter(init_params_tensor.clone().to(device))
    EQ_memory = 128 # TODO: hardcoded for now (should be greater than 0)

    # Load/synthesise the input audio (as torch tensors)
    if input_type == "white_noise":
        # Generate white noise excitation
        T = int(max_audio_len_s * sr)
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
    input = input.view(1,1,-1).to(device)

    # Allocate results buffers
    y_control = torch.zeros(1,1,T, device=device)

    # Initialize buffers
    in_buffer = torch.zeros(1,1,frame_len, device=device)
    EQ_out_len = next_power_of_2(frame_len + EQ_memory - 1)
    EQ_out_buffer = torch.zeros(1,1,EQ_out_len, device=device)
    LEM_out_len = frame_len + LEM_memory - 1
    LEM_out_buffer = torch.zeros(1,1,LEM_out_len, device=device)

    # Hanning window for overlap-add (use 50% overlap for perfect reconstruction)
    window = torch.hann_window(frame_len, periodic=True, device=device).view(1, 1, -1)

    # Set optimization (adaptive filtering)
    mu = mu_cont / frame_len # TODO: check step size normalization carefully!
    optimizer = torch.optim.SGD([EQ_params], lr=mu)  # lr is set to 1.0 because we manually scale the gradient
    
    # Build desired response: delay + optional target magnitude response
    total_delay = lem_delay+7  # TODO: add EQ group delay if necessary
    desired_response = build_desired_response_lin_phase(
        sr=sr,
        response_type=desired_response_type,
        target_mag_resp=target_mag_resp,
        target_mag_freqs=target_mag_freqs,
        fir_len=1024,
        ROI=ROI,
        rolloff_octaves=1.0,
        device=device
    )
    
    # Convert linear-phase desired response to minimum phase
    # This preserves magnitude response but minimizes group delay
    h_linear_np = desired_response.squeeze().cpu().numpy()
    h_minphase_np = minimum_phase(h_linear_np, method='homomorphic', half=False)
    
    # Add delay by prepending zeros to the minimum-phase filter
    delay_zeros = torch.zeros(total_delay, device=device)
    h_minphase = torch.from_numpy(h_minphase_np).float().to(device)
    desired_response = torch.cat([delay_zeros, h_minphase]).view(1, 1, -1)
    print(f"Minimum-phase desired response: {desired_response.shape[-1]} samples (delay={total_delay})")
    print(f"Converted to minimum phase: {desired_response.shape[-1]} samples")
    
    # Precompute desired output by convolving input with desired response
    # Use "full" mode then trim to match input length
    desired_output = torchaudio.functional.fftconvolve(input, desired_response, mode="full")
    print(f"Precomputed desired output (type: {desired_response_type})")
    
    # Initialize loss history for logging
    loss_history = []
    
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
        # Extract target frame from precomputed desired_output
        target_frame = desired_output[:, :, start_idx:start_idx + frame_len]
        loss = F.mse_loss(LEM_out_buffer[:, :, :frame_len], target_frame)
        loss_history.append(loss.item())

        #frame_analysis_plot(in_buffer, EQ_out_buffer[:,:,:frame_len], LEM_out_buffer[:, :, :frame_len], target_frame, frame_idx=k)

        # Backpropagate and update EQ parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            EQ_params.clamp_(0.0, 1.0)
            # Detach buffers to prevent graph accumulation across iterations
            EQ_out_buffer = EQ_out_buffer.detach()
            LEM_out_buffer = LEM_out_buffer.detach()

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
    
    desired_output_save = desired_output.squeeze()[:T]  # Trim to input length
    desired_output_save = desired_output_save / (desired_output_save.abs().max() + 1e-10) * 0.9
    
    # Save audio files
    torchaudio.save(output_dir / "input.wav", input_save.unsqueeze(0).cpu(), sr)
    torchaudio.save(output_dir / "output_controlled.wav", y_control_save.unsqueeze(0).cpu().detach(), sr)
    torchaudio.save(output_dir / "desired_output.wav", desired_output_save.unsqueeze(0).cpu(), sr)
    
    print(f"\nAudio files saved to {output_dir}:")
    print(f"  - input.wav: Input signal")
    print(f"  - output_controlled.wav: Output after EQ and LEM")
    print(f"  - desired_output.wav: Target/desired output signal")

    # Plot loss progression
    plt.figure(figsize=(12, 4))
    time_axis = np.arange(len(loss_history)) * hop_len / sr
    plt.semilogy(time_axis, loss_history, linewidth=1)
    plt.xlabel("Time (s)")
    plt.ylabel("MSE Loss")
    plt.title("Loss Progression During Adaptation")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()