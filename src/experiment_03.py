import sys
from pathlib import Path
from unittest import case
import numpy as np
import matplotlib.pyplot as plt
import torch, torchaudio
import torch.nn.functional as F
from torch.linalg import lstsq
from torch.func import jacrev, jacfwd
from scipy.signal import firls, freqz, minimum_phase
from tqdm import tqdm
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "lib"))
from lib.local_dasp_pytorch.modules import ParametricEQ
from modules import Ridge
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
        desired_response = torch.zeros(1, 1, fir_len, device=device)
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



def params_to_loss(EQ_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_response_buffer,
            EQ,
            LEM,
            frame_len,
            hop_len,
            target_frame,
            desired_response,
            forget_factor,
            loss_fcn,
            loss_type,
            sr=None,
            ROI=None):
    # Process through EQ
    EQ_out = EQ.process_normalized(in_buffer, EQ_params)

    # Update EQ output buffer (shift left by hop_len and add new samples)
    EQ_out_buffer = F.pad(EQ_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    EQ_out_buffer += EQ_out

    # Process through LEM
    LEM_out = torchaudio.functional.fftconvolve(EQ_out_buffer[:,:,:frame_len], LEM.view(1,1,-1), mode="full")
    
    # Update LEM output buffer (shift left by hop_len)
    LEM_out_buffer = F.pad(LEM_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    LEM_out_buffer += LEM_out
    
    # Use LEM output to compute loss and update EQ parameters
    match loss_type:
        case "FD-MSE" | "FD-SE":

            # # Deconvolve actual response within ROI limits
            nfft = 2*frame_len
            freqs = torch.fft.rfftfreq(nfft, d=1.0/sr, device=LEM_out_buffer.device)
            Y = torch.fft.rfft(LEM_out_buffer[:, :, :frame_len].squeeze(),n=nfft)      # Complex spectrum
            X = torch.fft.rfft(in_buffer.squeeze(), n=nfft)  # Complex spectrum
            eps = 1e-8
            H = Y / (X + eps)
            if ROI is not None:
                roi_mask = (freqs >= ROI[0]) & (freqs <= ROI[1])
                # Set H_complex to 1 (no amplification) outside ROI
                H = torch.where(roi_mask, H, torch.zeros_like(H) + eps)
            else:
                roi_mask = torch.ones_like(H, dtype=torch.bool)
            
            if torch.sum(torch.abs(est_response_buffer)) == 0:
                forget_factor = 1.0
            else:
                forget_factor = forget_factor
            H_mag_db_current = 20*torch.log10(torch.abs(H) + eps)
            H_mag_db = (forget_factor)*H_mag_db_current + (1-forget_factor)*est_response_buffer.squeeze()
            est_response_buffer = H_mag_db.view(1,1,-1).detach()
            desired_mag_db = 20*torch.log10(torch.abs(torch.fft.rfft(desired_response.squeeze(), n=nfft)) + eps)
            
            loss = loss_fcn(H_mag_db[roi_mask], desired_mag_db[roi_mask]) # Magnitude response MSE in log scale
            
        case _:
            loss = loss_fcn(LEM_out_buffer[:, :, :frame_len], target_frame)

    return loss



def process_buffers(EQ_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_response_buffer,
            EQ,
            LEM,
            frame_len,
            hop_len,
            target_frame,
            desired_response,
            forget_factor,
            loss_type,
            loss_fcn,
            sr=None,
            ROI=None,
            debug_plot_state=None):
    # Process through EQ
    EQ_out = EQ.process_normalized(in_buffer, EQ_params)

    # Update EQ output buffer (shift left by hop_len and add new samples)
    EQ_out_buffer = F.pad(EQ_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    EQ_out_buffer += EQ_out

    # Process through LEM
    LEM_out = torchaudio.functional.fftconvolve(EQ_out_buffer[:,:,:frame_len], LEM.view(1,1,-1), mode="full")
    
    # Update LEM output buffer (shift left by hop_len)
    LEM_out_buffer = F.pad(LEM_out_buffer[..., hop_len:], (0, hop_len))  # Shift buffer left
    LEM_out_buffer += LEM_out

    # Use LEM output to compute loss and update EQ parameters
    match loss_type:
        case "FD-MSE" | "FD-SE":

            # # Deconvolve actual response within ROI limits
            nfft = 2*frame_len-1
            freqs = torch.fft.rfftfreq(nfft, d=1.0/sr, device=LEM_out_buffer.device)
            Y = torch.fft.rfft(LEM_out_buffer[:, :, :frame_len].squeeze(),n=nfft)      # Complex spectrum
            X = torch.fft.rfft(in_buffer.squeeze(), n=nfft)  # Complex spectrum
            eps = 1e-8
            H = Y / (X + eps)
            if ROI is not None:
                roi_mask = (freqs >= ROI[0]) & (freqs <= ROI[1])
                # Set H_complex to 1 (no amplification) outside ROI
                H = torch.where(roi_mask, H, torch.zeros_like(H) + eps)
            else:
                roi_mask = torch.ones_like(H, dtype=torch.bool)
            
            if torch.sum(torch.abs(est_response_buffer)) == 0:
                forget_factor = 1.0
            else:
                forget_factor = forget_factor
            H_mag_db_current = 20*torch.log10(torch.abs(H) + eps)
            H_mag_db = (forget_factor)*H_mag_db_current + (1-forget_factor)*est_response_buffer.squeeze()
            est_response_buffer = H_mag_db.view(1,1,-1).detach()
            desired_mag_db = 20*torch.log10(torch.abs(torch.fft.rfft(desired_response.squeeze(), n=nfft)) + eps)
            
            loss = loss_fcn(H_mag_db[roi_mask], desired_mag_db[roi_mask]) # Magnitude response MSE in log scale

            if debug_plot_state is not None:
                freqs_roi = freqs[roi_mask].detach().cpu().numpy()
                H_mag_db_roi = H_mag_db[roi_mask].detach().cpu().numpy()
                
                # Smoothed version of actual magnitude response (moving average)
                smooth_window = 31  # Must be odd
                H_mag_db_roi_smoothed = np.convolve(H_mag_db_roi, np.ones(smooth_window)/smooth_window, mode='same')
                
                # Initialize plot on first call
                if debug_plot_state.get('fig') is None:
                    plt.ion()  # Enable interactive mode
                    fig, ax = plt.subplots(figsize=(12, 6))
                    line_raw, = ax.plot(freqs_roi, H_mag_db_roi, linewidth=0.5, alpha=0.4, label='Actual H(f)', color='tab:blue')
                    line_smooth, = ax.plot(freqs_roi, H_mag_db_roi_smoothed, linewidth=1.5, label='Actual H(f) (smoothed)', color='tab:blue')
                    line_desired, = ax.plot(freqs_roi, desired_mag_db[roi_mask].detach().cpu().numpy(), linewidth=1, label='Desired H(f)', color='tab:orange')
                    ax.set_xlabel("Frequency (Hz)")
                    ax.set_ylabel("Magnitude (dB)")
                    ax.set_xscale('log')
                    ax.set_title("FD-MSE: Actual vs Desired Magnitude Response")
                    ax.legend(loc='lower left')
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(-20, 30)  # Fixed y-axis range
                    plt.tight_layout()
                    debug_plot_state['fig'] = fig
                    debug_plot_state['ax'] = ax
                    debug_plot_state['line_raw'] = line_raw
                    debug_plot_state['line_smooth'] = line_smooth
                    debug_plot_state['line_desired'] = line_desired
                else:
                    # Update existing lines
                    debug_plot_state['line_raw'].set_ydata(H_mag_db_roi)
                    debug_plot_state['line_smooth'].set_ydata(H_mag_db_roi_smoothed)
                    debug_plot_state['line_desired'].set_ydata(desired_mag_db[roi_mask].detach().cpu().numpy())
                debug_plot_state['fig'].canvas.draw()
                debug_plot_state['fig'].canvas.flush_events()
            
        case _:
            loss = loss_fcn(LEM_out_buffer[:, :, :frame_len], target_frame)

    buffers = (EQ_out_buffer, LEM_out_buffer, est_response_buffer)

    return loss, buffers



def squared_error(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute element-wise squared error between predicted and true signals.
    
    Args:
        y_pred: Predicted signal tensor
        y_true: True signal tensor
    Returns:
        Element-wise squared error tensor
    """
    return (y_pred - y_true) ** 2




#%% MAIN SCRIPT

if __name__ == "__main__":

    torch.manual_seed(123)                  # Seed for reproducibility

    # Set paths
    base = Path(".")
    rir_dir = base / "data" / "rir"
    audio_input_dir = base / "data" / "audio" / "input"

    # Input configuration
    input_type = "white_noise"              # Either a file or a valid synthesisable signal
    max_audio_len_s = 15.0                  # None = full length

    # Simulation configuration
    ROI = [100.0, 12000.0]                  # region of interest for EQ compensation (Hz)
    frame_len = 1024*2                      # Length (samples) of processing buffers
    hop_len = frame_len                     # Stride between frames
    window_type = None                      # "hann" or None
    forget_factor = 0.1                     # Forgetting factor for FD loss estimation (0=no memory, 1=full memory)
    optim_type = "GHAM-1"                   # "SGD", "Adam", "LBFGS", "GHAM-1" or "Muon" TODO get newer PyTorch for Muon
    mu_opt = 0.01#*1e-1                      # Learning rate for controller (*1e3  Adam) (*1e-2  SGD) (*1e0 GHAM-1)
    loss_type = "FD-MSE"                    # "TD-MSE", "FD-MSE", "TD-SE"
    desired_response_type = "delay_and_mag" # "delay_and_mag" or "delay_only"
    scenario_type = "sudden"              # "constant", "sudden" or "smooth" (not implemented yet)
    n_rirs = 2                              # Number of RIRs to simulate (for time-varying scenarios)
    debug_plot_state = {}                   # Debug plot state (set to None to disable, or {} to enable)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Acoustic path from actuator (speaker) to sensor (microphone)
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs)
    match scenario_type:
        case "constant":
            rir_init = rirs[0]
            sr = rirs_srs[0]
        case "sudden":
            # TODO: implement sudden scenario
            rir_init = rirs[0]
            sr = rirs_srs[0]
            switch_times_norm = [t/n_rirs for t in range(1,n_rirs)]
        case _:
            raise NotImplementedError(f"Scenario type '{scenario_type}' not implemented yet.")

    # Desired response computation and initial EQ parameters
    lem_delay = get_delay_from_ir(rir_init, sr)
    EQ_comp_dict = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=6)
    target_mag_resp = EQ_comp_dict["target_response_db"]
    target_mag_freqs = EQ_comp_dict["freq_axis_smoothed"]
    # TODO: interpolate to actual FFT bin used later (frame_len)

    # Initialize the LEM estimate (assume LEM is well-identified)
    LEM = torch.from_numpy(rir_init).view(1,1,-1).to(device)
    LEM_memory = LEM.shape[-1]

    # Initialize differentiable EQ
    EQ = ParametricEQ(sample_rate=sr)
    #init_params_tensor = torch.rand(1,EQ.num_params) # random initialization: It's pretty sensitive to initial parameters
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
        input = torch.randn(T, device=device)
        print(f"Input signal: White noise ({max_audio_len_s} s, {T} samples)")

    else:
        # Load audio file from data/audio/input/
        audio_path = audio_input_dir / input_type
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        input, audio_sr = torchaudio.load(audio_path)
        input = input.to(device)
        
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

    # Hanning window for overlap-add (use 50% overlap for perfect reconstruction)
    match window_type:
        case "hann":
            cond = frame_len / (2*hop_len) 
            assert cond.is_integer() and int(cond) > 0, "Invalid hop_length for Hann window. Must have frame_len / (2*hop_len) = integer > 0."
            window = torch.hann_window(frame_len, periodic=True, device=device).view(1, 1, -1)
        case _:
            cond = frame_len / hop_len
            assert cond.is_integer() and (cond % 2 == 0 or cond == 1), "Invalid hop_length for rectangular window."
            window = torch.ones(1, 1, frame_len, device=device)

    # Set optimization (adaptive filtering)
    match optim_type:
        case "SGD":
            mu = mu_opt # TODO: check step size normalization carefully!
            optimizer = torch.optim.SGD([EQ_params], lr=mu)
        case "Adam":
            mu = mu_opt / frame_len # TODO: check step size normalization carefully!
            optimizer = torch.optim.Adam([EQ_params], lr=mu)
        case "Muon":
            raise ValueError("Muon optimizer requires newer PyTorch version.")
            mu = mu_opt / frame_len # TODO: check step size normalization carefully!
            optimizer = torch.optim.Muon([EQ_params], lr=mu_opt)
        case "LBFGS":
            raise ValueError("LBFGS optimizer requires multiple function evaluations per optimization step. Not suitable for adaptive filtering scenario.")
        case "GHAM-1":
            mu = mu_opt # TODO: check step size normalization carefully!
            eps_0 = 20 # Irreducible error floor
            optimizer = None # No optimizer object needed yet! TODO
            alpha_ridge = 1e-3
            ridge_regressor = Ridge(alpha = alpha_ridge, fit_intercept = False)

    # Build desired response: delay + optional target magnitude response
    total_delay = lem_delay + 7  # TODO: add EQ group delay if necessary. Hardcoded for now!
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
    
    # Convert linear-phase desired response to minimum phase (minimize group delay) and add desired delay
    h_linear_np = desired_response.squeeze().cpu().numpy()
    h_minphase_np = minimum_phase(h_linear_np, method='homomorphic', half=False)
    delay_zeros = torch.zeros(total_delay, device=device)
    h_minphase = torch.from_numpy(h_minphase_np).float().to(device)
    desired_response = torch.cat([delay_zeros, h_minphase]).view(1, 1, -1)

    # Precompute desired output
    desired_output = torchaudio.functional.fftconvolve(input, desired_response, mode="full")
    print(f"Precomputed desired output (type: {desired_response_type})")
    
    # Initialize loss & loss history
    match loss_type:
        case "TD-MSE" | "FD-MSE":
            loss_fcn = F.mse_loss
        case "TD-SE" | "FD-SE":
            loss_fcn = squared_error
        case _:
            raise NotImplementedError(f"Not yet implemented loss_type: {loss_type}. Use 'TD-MSE'")
    loss_history = []
    jac_norm_history = []
    jac_cond_history = []
    irreducible_loss_history = []

    # Initialize buffers
    in_buffer = torch.zeros(1,1,frame_len, device=device)
    EQ_out_len = next_power_of_2(2*frame_len - 1) # match dasp_pytorch sosfilt_via_fms implementation
    EQ_out_buffer = torch.zeros(1,1,EQ_out_len, device=device)
    LEM_out_len = frame_len + LEM_memory - 1
    LEM_out_buffer = torch.zeros(1,1,LEM_out_len, device=device)
    est_response_buffer = torch.zeros(1,1,frame_len, device=device) # TODO stabilize response estimation
    rir_idx = 0
    
    # Main loop
    n_frames = (T - frame_len) // hop_len + 1
    for k in tqdm(range(n_frames), desc="Processing", unit="frame"):

        start_idx = k * hop_len

        # find first instance where start_idx/T is greater than a switch time
        match scenario_type:
            case "sudden":
                current_idx = 0 
                for switch_time in switch_times_norm:
                    if (start_idx / T) >= switch_time:
                        current_idx += 1
                    if current_idx > rir_idx:
                        rir_idx = current_idx
                        LEM = torch.from_numpy(rirs[current_idx]).view(1,1,-1).to(device)
        
        # Update input buffer and apply window
        in_buffer = input[:,:,start_idx:start_idx+frame_len] * window

        # Get target frame
        target_frame = desired_output[:, :, start_idx:start_idx + frame_len]

        # I think jacrev (and jacfwd) forward the function to differentiate each time, so it's inefficient here.
        #grad_fcn = jacrev(params_to_loss, argnums=0, has_aux=False)
        #hess_fcn = jacfwd(grad_fcn, argnums=0, has_aux=False)
        #jac3_fcn = jacfwd(hess_fcn, argnums=0, has_aux=False)
        #jac4_fcn = jacfwd(jac3_fcn, argnums=0, has_aux=False)
        #grad = grad_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,EQ,LEM,frame_len,hop_len,target_frame,loss_fcn).squeeze()
        #hess = hess_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,EQ,LEM,frame_len,hop_len,target_frame,loss_fcn).squeeze()
        #jac3 = jac3_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,EQ,LEM,frame_len,hop_len,target_frame,loss_fcn).squeeze()
        #jac4 = jac4_fcn(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,EQ,LEM,frame_len,hop_len,target_frame,loss_fcn).squeeze()

        loss, buffers = process_buffers(EQ_params,
            in_buffer,
            EQ_out_buffer,
            LEM_out_buffer,
            est_response_buffer,
            EQ,
            LEM,
            frame_len,
            hop_len,
            target_frame,
            desired_response,
            forget_factor,
            loss_type,
            loss_fcn,
            sr=sr,
            ROI=ROI,
            debug_plot_state=debug_plot_state)
        EQ_out_buffer, LEM_out_buffer, est_response_buffer = buffers

        loss_history.append(torch.mean(loss).item())

        #frame_analysis_plot(in_buffer, EQ_out_buffer[:,:,:frame_len], LEM_out_buffer[:, :, :frame_len], target_frame, frame_idx=k)

        # Backpropagate and update EQ parameters
        match optim_type:
            case "GHAM-1":
                match loss_type:
                    case "TD-MSE" | "FD-MSE":
                        loss.backward()
                        jac = EQ_params.grad.clone().view(1,-1)
                    case "TD-SE" | "FD-SE":
                        jac = jacfwd(params_to_loss, argnums=0, has_aux=False)(EQ_params,in_buffer,EQ_out_buffer,LEM_out_buffer,est_response_buffer,EQ,LEM,frame_len,hop_len,target_frame,desired_response,forget_factor,loss_fcn,loss_type,sr=None,ROI=None).squeeze()
                
                loss_val = loss.detach() - torch.tensor(eps_0, device=device)
                
                # Log irreducible loss and jacobian condition number
                irreducible_loss_history.append(loss_val.mean().item())
                jac_cond_history.append(torch.linalg.cond(jac.detach().cpu().float()).item())
                
                with torch.no_grad():
                    b = loss_val.view(-1,1)                # (loss_dims, 1)
                    update = lstsq(jac, b).solution        # (num_params, 1)
                    #ridge_regressor.fit(jac,b)
                    #update_ridge = ridge_regressor.w      # (num_params, 1)
                    EQ_params -= mu * update.view_as(EQ_params)
                EQ_params.grad = None

            case _:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        with torch.no_grad():
            EQ_params.clamp_(0.0, 1.0)
            EQ_out_buffer = EQ_out_buffer.detach() # prevent graph accumulation across iterations
            LEM_out_buffer = LEM_out_buffer.detach()
            est_response_buffer = est_response_buffer.detach()

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

    # Plot loss progression (2x1 subplot: log scale on top, linear scale on bottom)
    fig, (ax_log, ax_lin) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    time_axis = np.arange(len(loss_history)) * hop_len / sr
    
    # ---- Top subplot: Log scale ----
    # Left axis: Loss and Irreducible Loss
    ax_log.semilogy(time_axis, loss_history, linewidth=1, label='Loss', color='tab:blue')
    if irreducible_loss_history:
        time_axis_irr = np.arange(len(irreducible_loss_history)) * hop_len / sr
        ax_log.semilogy(time_axis_irr, irreducible_loss_history, linewidth=1, label='Irreducible Loss', color='tab:orange')
    ax_log.set_ylabel("Loss (log)")
    ax_log.grid(True, alpha=0.3)
    ax_log.legend(loc='upper left')
    ax_log.set_title("Loss Progression During Adaptation (Log Scale)")
    
    # Right axis: Jacobian condition number (log)
    if jac_cond_history:
        ax_log_r = ax_log.twinx()
        time_axis_cond = np.arange(len(jac_cond_history)) * hop_len / sr
        ax_log_r.semilogy(time_axis_cond, jac_cond_history, linewidth=1, label='Jacobian Cond. Number', color='tab:green')
        ax_log_r.set_ylabel("Condition Number", color='tab:green')
        ax_log_r.tick_params(axis='y', labelcolor='tab:green')
        ax_log_r.legend(loc='upper right')
    
    # ---- Bottom subplot: Linear scale ----
    # Left axis: Loss and Irreducible Loss
    ax_lin.plot(time_axis, loss_history, linewidth=1, label='Loss', color='tab:blue')
    if irreducible_loss_history:
        time_axis_irr = np.arange(len(irreducible_loss_history)) * hop_len / sr
        ax_lin.plot(time_axis_irr, irreducible_loss_history, linewidth=1, label='Irreducible Loss', color='tab:orange')
    ax_lin.set_xlabel("Time (s)")
    ax_lin.set_ylabel("Loss (linear)")
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend(loc='upper left')
    ax_lin.set_title("Loss Progression During Adaptation (Linear Scale)")
    
    # Right axis: Jacobian condition number (linear)
    if jac_cond_history:
        ax_lin_r = ax_lin.twinx()
        time_axis_cond = np.arange(len(jac_cond_history)) * hop_len / sr
        ax_lin_r.plot(time_axis_cond, jac_cond_history, linewidth=1, label='Jacobian Cond. Number', color='tab:green')
        ax_lin_r.set_ylabel("Condition Number", color='tab:green')
        ax_lin_r.tick_params(axis='y', labelcolor='tab:green')
        ax_lin_r.legend(loc='upper right')
    
    plt.tight_layout()
    plt.ioff()  # Turn off interactive mode so plt.show() blocks
    plt.show()  # This will block until all figures are closed

    pass