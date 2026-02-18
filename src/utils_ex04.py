import json, itertools, random, sys, torch
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import soundfile as sf
import torch.nn.functional as F
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

root = Path(__file__).resolve().parent.parent

##############################
# Signal processing utiities #
##############################

def get_delay_from_ir(rir: np.ndarray, sr: int) -> int:
    """Estimate delay (in samples) from an RIR by finding the position of the first maximum.

    Finds the sample index where the first maximum (peak) of the RIR occurs.
    This is a robust estimate of the direct sound arrival time.

    Args:
        rir: 1-D numpy array containing the RIR
        sr: sample rate (Hz)
    Returns:
        delay_samples: estimated delay in samples (int) - index of the maximum value
    """
    # Find the index of the maximum absolute value
    delay_samples = np.argmax(np.abs(rir))
    return delay_samples


def _six_biquad_eq_params_to_dict(EQ_params: np.ndarray) -> dict:
    """
    Convert (6, 3) EQ parameter matrix into a named parameter dictionary.

    EQ_params: np.ndarray with shape (6, 3)
               columns: [gain_dB, Q_or_slope, Fc]
    """
    assert EQ_params.shape == (6, 3), "Expected EQ_params shape (6, 3)"

    return {
        # ---- Low shelf ----
        "low_shelf_gain_db": float(EQ_params[0, 0]),
        "low_shelf_q_factor": float(EQ_params[0, 1]),
        "low_shelf_cutoff_freq": float(EQ_params[0, 2]),

        # ---- Band 0 ----
        "band0_gain_db": float(EQ_params[1, 0]),
        "band0_q_factor": float(EQ_params[1, 1]),
        "band0_cutoff_freq": float(EQ_params[1, 2]),

        # ---- Band 1 ----
        "band1_gain_db": float(EQ_params[2, 0]),
        "band1_q_factor": float(EQ_params[2, 1]),
        "band1_cutoff_freq": float(EQ_params[2, 2]),

        # ---- Band 2 ----
        "band2_gain_db": float(EQ_params[3, 0]),
        "band2_q_factor": float(EQ_params[3, 1]),
        "band2_cutoff_freq": float(EQ_params[3, 2]),

        # ---- Band 3 ----
        "band3_gain_db": float(EQ_params[4, 0]),
        "band3_q_factor": float(EQ_params[4, 1]),
        "band3_cutoff_freq": float(EQ_params[4, 2]),

        # ---- High shelf ----
        "high_shelf_gain_db": float(EQ_params[5, 0]),
        "high_shelf_q_factor": float(EQ_params[5, 1]),
        "high_shelf_cutoff_freq": float(EQ_params[5, 2]),
    }


def biquad_coefficients(
    gain_db: float,
    fc: float,
    Q: float,
    Fs: float,
    filter_type: str,
):
    """
    RBJ cookbook biquad coefficients.
    Returns (b, a) normalized so a[0] = 1.
    """
    A = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * fc / Fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2 * Q)

    if filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

    elif filter_type == "low_shelf":
        sqrtA = np.sqrt(A)
        two_sqrtA_alpha = 2 * sqrtA * alpha
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha

    elif filter_type == "high_shelf":
        sqrtA = np.sqrt(A)
        two_sqrtA_alpha = 2 * sqrtA * alpha
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + two_sqrtA_alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - two_sqrtA_alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + two_sqrtA_alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - two_sqrtA_alpha

    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # Normalize
    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0

    return b, a


def compute_parametric_eq_response(
    params: np.ndarray,
    f: np.ndarray,
    Fs: float,
) -> np.ndarray:
    """
    Compute total EQ filter response in dB using SOS biquad cascade.

    params: (num_filters, 3) -> [gain_dB, Q_or_slope, Fc]
    f: frequency vector in Hz
    """
    num_filters = params.shape[0]

    # Build SOS array: (num_filters, 6)
    sos = np.zeros((num_filters, 6))

    for i, (gain_db, Q, Fc) in enumerate(params):
        if i == 0:
            ftype = "low_shelf"
        elif i == num_filters - 1:
            ftype = "high_shelf"
        else:
            ftype = "peaking"

        b, a = biquad_coefficients(
            gain_db=gain_db,
            fc=Fc,
            Q=Q,
            Fs=Fs,
            filter_type=ftype,
        )

        sos[i, :] = np.hstack([b, a])

    # Convert frequency vector to digital radian frequency
    worN = 2 * np.pi * f / Fs

    # Complex frequency response
    H = sosfreqz_np(sos, worN)

    # Magnitude in dB
    filt_resp_db = 20.0 * np.log10(np.abs(H) + 1e-12)

    return filt_resp_db


def sosfreqz_np(sos: np.ndarray, worN: np.ndarray) -> np.ndarray:
    """
    Compute complex frequency response of SOS cascade.

    sos: (n_sections, 6) → [b0 b1 b2 a0 a1 a2]
    worN: digital radian frequencies
    """
    z = np.exp(1j * worN)
    H = np.ones_like(z, dtype=np.complex128)

    for sec in sos:
        b0, b1, b2, a0, a1, a2 = sec
        num = b0 + b1 / z + b2 / z**2
        den = a0 + a1 / z + a2 / z**2
        H *= num / den

    return H


def _eq_objective_function(
    params_flat: np.ndarray,
    num_filters: int,
    f: np.ndarray,
    meas_resp_db: np.ndarray,
    target_resp_db: np.ndarray,
    ROI: Tuple[float, float],
    Fs: float,
) -> np.ndarray:
    """
    Residual vector for least-squares EQ optimization.
    """
    params = params_flat.reshape(num_filters, 3)

    filt_resp_db = compute_parametric_eq_response(params, f, Fs)
    output_resp_db = meas_resp_db + filt_resp_db

    # Restrict error to ROI
    idx = (f >= ROI[0]) & (f <= ROI[1])
    error = output_resp_db[idx] - target_resp_db[idx]

    return error


def _get_initial_gains(n: int, seed: int = 0) -> np.ndarray:
    """
    Generate initial gain values for EQ filters.
    Uses a fixed RNG seed so results are repeatable.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


def _init_eq_parameters(
    num_filters: int,
    ROI: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize EQ parameter matrix init_params and bounds lb / ub.

    Parameter format per filter:
        [gain_dB, Q_or_slope, Fc]
    """
    fmin = max(20.0, ROI[0])
    fmax = min(20e3, ROI[1])

    # Center frequencies
    Fc_peaks = np.logspace(                 # All but last two are peaking filters
        np.log10(fmin),
        np.log10(fmax),
        num_filters - 2
    )
    Fc_shelves = np.array([                 # Last two filters are shelves (high and low)
        (2 * ROI[0] + ROI[1]) / 3,
        (ROI[0] + 2 * ROI[1]) / 3
    ])

    # Initial parameter matrix
    init_params = np.zeros((num_filters, 3))
    init_params[:, 0] = _get_initial_gains(num_filters)
    init_params[:-2, 1] = 1.5               # Q for peak filters
    init_params[-2:, 1] = 1.0               # slope for shelf filters
    init_params[:-2, 2] = Fc_peaks
    init_params[-2:, 2] = Fc_shelves

    # Lower bounds
    lb = np.zeros_like(init_params)
    lb[:, 0] = -24.0                        # gain lower bound
    lb[:-2, 1] = 0.2                        # Q for peak filters lower bound
    lb[-2:, 1] = 0.1                        # slope for shelf filters lower bound
    lb[:, 2] = fmin-1e-6                    # Fc lower bound

    # Upper bounds
    ub = np.zeros_like(init_params)
    ub[:, 0] = 20.0                         # gain upper bound
    ub[:-2, 1] = 17.3                       # Q for peak filters upper bound
    ub[-2:, 1] = 5.0                        # slope for shelf filters upper bound
    ub[:, 2] = fmax+1e-6                    # Fc upper bound

    return init_params, lb, ub


def _eq_optimizer(
    num_filters: int,
    f: np.ndarray,
    meas_resp_db: np.ndarray,
    target_resp_db: np.ndarray,
    ROI: Tuple[float, float],
    Fs: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize parametric EQ parameters.

    Returns:
        EQ         -> optimized parameter matrix (num_filters, 3)
        outputResp -> final equalized response (dB)
        filtResp   -> EQ filter response (dB)
    """
    init_params, lb, ub = _init_eq_parameters(num_filters, ROI)

    res = least_squares(
        _eq_objective_function,                                         # loss
        init_params.ravel(),                                            # initial parameters (flattened)
        bounds=(lb.ravel(), ub.ravel()),
        args=(num_filters, f, meas_resp_db, target_resp_db, ROI, Fs),   # additional args to compute loss function
        ftol=1e-8,
        max_nfev=min(500 * num_filters, 1000),
        verbose=1,
    )

    EQ_params = res.x.reshape(num_filters, 3)                           # Extract optimized (LS) EQ parameters
    filtResp = compute_parametric_eq_response(EQ_params, f, Fs)
    outputResp = meas_resp_db + filtResp

    return EQ_params, outputResp, filtResp


def _get_target_response_comp_EQ(cf: np.ndarray, oa: np.ndarray, ROI: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the linearized target response (in dB) for compensation EQ based on octave-averaged response.

    Args:
        cf: center frequencies (Hz) for octave bands (1D array)
        oa: octave-averaged magnitudes (linear) corresponding to cf (1D array)
        ROI: region of interest [fmin, fmax] (Hz)

    Returns:
        target_resp: target response in dB (same length as cf)
        pfit: array([intercept, slope]) of the linear fit in log-frequency domain (dB = intercept + slope * log(freq))
        pdb: octave-averaged response in dB (same length as cf)
    """
    # Defensive copy / ensure arrays
    cf = np.atleast_1d(cf).astype(float)
    oa = np.atleast_1d(oa).astype(float)

    # Convert to dB (avoid log(0))
    pdb = 20.0 * np.log10(oa + 1e-12)

    # Set the range of the linear fit (translated from MATLAB)
    lfCutOff = 2.5 * ROI[0]   # lowest frequency to linearize
    hfMaxFit = 0.6 * ROI[1]   # max frequency to fit for

    # Only consider positive center freqs for log
    valid_cf = cf > 0.0
    linIdx = valid_cf & (cf >= lfCutOff) & (cf <= hfMaxFit)

    # Initialize outputs
    pfit = np.array([0.0, 0.0], dtype=float)
    target_resp = np.copy(pdb)

    if np.any(linIdx):
        # Fit straight line in log-frequency domain: pdb = intercept + slope * log(cf)
        logx = np.log(cf[linIdx])
        y = pdb[linIdx]

        # Simple linear regression via polyfit (slope first, intercept second)
        slope, intercept = np.polyfit(logx, y, 1)
        pfit = np.array([intercept, slope])  # MATLAB ordering: pfit(1)=intercept, pfit(2)=slope

        # Build target response across all cf
        target_resp = pfit[0] + pfit[1] * np.log(cf)

        # Roll off the low frequencies, starting slightly above the linear range:
        lfcutoff = 1.05 * lfCutOff
        idx_low = cf < lfcutoff
        if np.any(idx_low):
            # attenuation amount: min(30, ROI[0]/2) * ((lfcutoff - cf)/lfcutoff)^2
            max_atten = min(30.0, ROI[0] / 2.0)
            frac = (lfcutoff - cf[idx_low]) / lfcutoff
            rolloff = max_atten * (frac ** 2)
            target_resp[idx_low] = target_resp[idx_low] - rolloff
    else:
        # No suitable linear region: keep pdb as target and leave pfit zeros
        target_resp = np.copy(pdb)
        pfit = np.array([0.0, 0.0])

    return target_resp, pfit, pdb


def _octave_average(f: np.ndarray, resp: np.ndarray, bpo: int, 
                   freq_range: Tuple[float, float] = None, b_smooth: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute average power per fractional octave band.
    
    Based on MATLAB's octaveAverage function. Uses constant bandwidth below 
    300-400 Hz (Bark scale approximation) and octave spacing above.
    
    Args:
        f: Frequency vector (Hz)
        resp: Magnitude response (linear, not dB)
        bpo: Bands per octave (e.g., 1 for 1-octave, 3 for 1/3-octave, 6 for 1/6-octave)
        freq_range: Frequency range [fmin, fmax]. If None, uses full range
        b_smooth: Apply smoothing to output (recommended for noisy responses)
    
    Returns:
        oa: Averaged octave magnitudes
        cf: Center frequencies (Hz)
    """
    # Ensure row vectors
    f = np.atleast_1d(f).astype(float)
    resp = np.atleast_1d(resp).astype(float)
    
    # Apply frequency range filter
    if freq_range is not None:
        idx = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[idx]
        resp = resp[idx]
    
    # If range is empty
    if len(f) == 0:
        return np.array([]), np.array([])
    
    # Compute center frequencies (cf) and band-edge frequencies (bef)
    G = 10 ** (3/10)  # ~2.0 (octave ratio)
    ref_freq = f[0]
    f_range = np.array([f[0], f[-1]])
    lgbg = bpo / np.log(G)
    
    # Use a constant bandwidth below 400 Hz (based on Zwicker's Bark scale resolution)
    octave_cutoff = 300  # Hz, set between 120 and 400
    f_min = int(np.round(lgbg * np.log(octave_cutoff / ref_freq)))
    f_max = int(np.floor(lgbg * np.log(f_range[1] / ref_freq)))
    
    # Octave-spaced center frequencies above cutoff
    cf = ref_freq * (G ** (np.arange(f_min, f_max + 1) / bpo))
    
    # Constant bandwidth section below cutoff
    last_cst_cf = ref_freq * (G ** (f_min / bpo))
    lf_bw = ref_freq * (G ** (f_min / bpo) - G ** ((f_min - 1) / bpo))
    nb_cst = 1 + int(np.ceil((last_cst_cf - f_range[0]) / lf_bw))
    
    if nb_cst > 1:
        lf_cf = np.linspace(f_range[0], last_cst_cf, nb_cst)
        cf = np.concatenate([lf_cf[:-1], cf])
    
    # Band-edge frequencies
    bef = np.concatenate([cf[:-1] * (G ** (1 / (2 * bpo))), [f[-1]]])
    
    oa = np.full(len(cf), np.nan)
    
    # Average each octave band
    f_beg = 0
    for ii in range(len(cf)):
        f_end_indices = np.where(f <= bef[ii])[0]
        if len(f_end_indices) > 0:
            f_end = f_end_indices[-1] + 1
            oa[ii] = np.nanmean(resp[f_beg:f_end])
            f_beg = f_end
    
    # Handle empty bins (when octave bandwidth > available frequency points)
    idx = np.where(~np.isnan(oa[:-1]) & np.isnan(oa[1:]))[0]
    for ii in idx:
        oa_idx = ii
        f_end_indices = np.where(~np.isnan(oa[oa_idx + 1:]))[0]
        if len(f_end_indices) > 0:
            f_end = f_end_indices[0]
            # Geometric mean of center frequencies
            cf[oa_idx] = np.prod(cf[oa_idx:oa_idx + f_end]) ** (1 / f_end)
        else:
            break
    
    # Remove empty bins
    idx = ~np.isnan(oa)
    oa = oa[idx]
    cf = cf[idx]
    
    # Apply smoothing if requested (for measured/noisy responses)
    if b_smooth:
        # First smoothing pass (3-point average)
        oa[1:-1] = (1*oa[1:-1] + oa[:-2] + oa[2:]) / 3
        # Second smoothing pass (5-point average)
        if len(oa) > 4:
            oa[2:-2] = (2*oa[2:-2] + oa[:-4] + oa[4:]) / 4
    
    return oa, cf


def octave_average_torch(f: torch.Tensor, resp: torch.Tensor, bpo: int, 
                         freq_range: Tuple[float, float] = None, b_smooth: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute average power per fractional octave band (PyTorch version).
    
    Preserves complex values (e.g., for phase information in frequency responses).
    Works with PyTorch tensors on GPU/CPU.
    
    Based on MATLAB's octaveAverage function. Uses constant bandwidth below 
    300-400 Hz (Bark scale approximation) and octave spacing above.
    
    Args:
        f: Frequency vector (Hz) as torch tensor
        resp: Response (complex or real) as torch tensor
        bpo: Bands per octave (e.g., 1 for 1-octave, 3 for 1/3-octave, 6 for 1/6-octave)
        freq_range: Frequency range [fmin, fmax]. If None, uses full range
        b_smooth: Apply smoothing to output (recommended for noisy responses)
    
    Returns:
        resp_averaged: Averaged octave response (complex or real, depending on input)
        cf: Center frequencies (Hz)
    """
    device = f.device
    dtype = resp.dtype
    
    # Apply frequency range filter
    if freq_range is not None:
        mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f = f[mask]
        resp = resp[mask]
    
    # If range is empty
    if len(f) == 0:
        return torch.tensor([], device=device, dtype=dtype), torch.tensor([], device=device)
    
    # Compute center frequencies (cf) and band-edge frequencies (bef)
    G = 10 ** (3/10)  # ~2.0 (octave ratio)
    ref_freq = f[0].item()+1e-6
    f_min_val = f[0].item()
    f_max_val = f[-1].item()
    lgbg = bpo / np.log(G)
    
    # Use a constant bandwidth below 400 Hz (based on Zwicker's Bark scale resolution)
    octave_cutoff = 300  # Hz, set between 120 and 400
    f_min = int(np.round(lgbg * np.log(octave_cutoff / ref_freq)))
    f_max = int(np.floor(lgbg * np.log(f_max_val / ref_freq)))
    
    # Octave-spaced center frequencies above cutoff
    cf = ref_freq * (G ** (np.arange(f_min, f_max + 1) / bpo))
    cf = torch.tensor(cf, device=device, dtype=torch.float32)
    
    # Constant bandwidth section below cutoff
    last_cst_cf = ref_freq * (G ** (f_min / bpo))
    lf_bw = ref_freq * (G ** (f_min / bpo) - G ** ((f_min - 1) / bpo))
    nb_cst = 1 + int(np.ceil((last_cst_cf - f_min_val) / lf_bw))
    
    if nb_cst > 1:
        lf_cf = torch.linspace(f_min_val, last_cst_cf, nb_cst, device=device)
        cf = torch.cat([lf_cf[:-1], cf])
    
    # Band-edge frequencies
    bef = torch.cat([cf[:-1] * (G ** (1 / (2 * bpo))), torch.tensor([f_max_val], device=device)])
    
    resp_averaged = torch.full((len(cf),), torch.nan, device=device, dtype=dtype)
    
    # Average each octave band
    f_beg = 0
    for ii in range(len(cf)):
        f_end_indices = torch.where(f <= bef[ii])[0]
        if len(f_end_indices) > 0:
            f_end = f_end_indices[-1] + 1
            resp_averaged[ii] = torch.nanmean(resp[f_beg:f_end])
            f_beg = f_end
    
    # Remove empty bins
    idx = ~torch.isnan(resp_averaged)
    resp_averaged = resp_averaged[idx]
    cf = cf[idx]
    
    # Apply smoothing if requested
    if b_smooth and len(resp_averaged) > 4:
        # First smoothing pass (3-point average)
        resp_averaged[1:-1] = (1*resp_averaged[1:-1] + resp_averaged[:-2] + resp_averaged[2:]) / 3
        # Second smoothing pass (5-point average)
        resp_averaged[2:-2] = (2*resp_averaged[2:-2] + resp_averaged[:-4] + resp_averaged[4:]) / 4
    
    return resp_averaged, cf


def get_compensation_EQ_params(rir: np.ndarray, sr: int, ROI: Tuple[float, float]=(20.0, 20000.0), num_sections: int=6) -> dict:
    """Estimate Parametric EQ parameters to compensate for the given RIR.

    This function analyzes the RIR
    and compute suitable EQ parameters to flatten its frequency response.
    """
    nfft = len(rir)
    freq_response = np.abs(np.fft.rfft(rir, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1/sr)

    # Apply octave averaging (no smoothing for filter responses)
    oa, cf = _octave_average(freqs, freq_response, bpo=24, freq_range=ROI, b_smooth=False)

    # Compute target response for compensation EQ
    target_resp, pfit, pdb = _get_target_response_comp_EQ(cf, oa, ROI)

    # Log the optimization start
    print(f"\nStarting initial EQ optimization with {num_sections} filters over ROI {ROI[0]}-{ROI[1]} Hz...")

    # Optimize parametric EQ to match target response
    EQ_params, out_resp_db, filt_resp_db = _eq_optimizer(
    num_filters=num_sections,                               # 6 filters for dasp_pytorch parametric eq
    f=cf,
    meas_resp_db=pdb,
    target_resp_db=target_resp,
    ROI=ROI,
    Fs=sr,
    )

    # Optional: plot results and save figure
    plt.figure()
    plt.semilogx(freqs, 20*np.log10(freq_response + 1e-12), label="RIR Frequency Response")
    plt.semilogx(cf, 20*np.log10(oa + 1e-12), label="Octave-Averaged Response")
    plt.semilogx(cf, target_resp, label="Target Compensation Response", linestyle='--')
    plt.semilogx(cf, out_resp_db, label="Equalized Response", linestyle='-.')
    plt.semilogx(cf, filt_resp_db, label="EQ Filter Response", linestyle=':')
    plt.axvline(ROI[0], color="red", linestyle="--", label="ROI Limits" if ROI else "")
    plt.axvline(ROI[1], color="red", linestyle="--" if ROI else "")
    plt.legend(), plt.xlabel("Frequency (Hz)"), plt.ylabel("Magnitude (dB)")
    plt.title("Room Frequency Response and Target Compensation EQ")
    plt.savefig("RFR_compensation.png",dpi=150, bbox_inches='tight')

    # Build parameter dictionary for downstream use (e.g., PyTorch EQ)
    eq_param_dict = _six_biquad_eq_params_to_dict(EQ_params)

    return {
        "eq_params": eq_param_dict,
        "EQ_matrix": EQ_params,
        "measured_response_db": pdb,
        "frq_axis_full": freqs,
        "target_response_db": target_resp,
        "equalized_response_db": out_resp_db,
        "filter_response_db": filt_resp_db,
        "freq_axis_smoothed": cf,
    }


#########################
# Main experiment logic #
#########################

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
        # Normalize so that maximum absolute value is one
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak
        rirs.append(data)
        srs.append(sr)
    return rirs, srs


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load JSON configuration for experiment_04.

    The JSON is expected to contain:
      - "simulation_params": {param_name: [values, ...]}
      - "input": {"use_white_noise": bool, "use_songs_folder": bool, "max_audio_len_s": [values, ...]}
    """
    with config_path.open("r") as f:
        cfg = json.load(f)
    return cfg


def iter_param_grid(param_grid: Dict[str, Iterable[Any]]) -> Iterable[Dict[str, Any]]:
    """Yield all combinations from a simple parameter grid.

    param_grid is a dict mapping param name -> iterable of candidate values.
    """
    keys = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[k] for k in keys))
    for values in values_product:
        yield {k: v for k, v in zip(keys, values)}


def discover_input_signals(input_cfg: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Discover which input signals should be used.

    Returns a list of (mode, info) tuples where mode is one of:
      - "white_noise": info contains {"max_audio_len_s": float}
      - "song": info contains {"path": Path, "max_audio_len_s": float}
    """
    modes: List[Tuple[str, Dict[str, Any]]] = []

    use_white_noise = bool(input_cfg.get("use_white_noise", False))
    use_songs_folder = bool(input_cfg.get("use_songs_folder", False))
    max_len_list = input_cfg.get("max_audio_len_s", [None])
    max_num_songs = input_cfg.get("max_num_songs", None)

    # For now, take the first max_audio_len_s if multiple are given; it will
    # be combined with the simulation params grid separately.
    max_audio_len_s = max_len_list[0] if max_len_list else None

    if use_white_noise:
        modes.append(("white_noise", {"max_audio_len_s": max_audio_len_s}))

    if use_songs_folder:
        songs_dir = root / "data" / "audio" / "input" / "songs"
        if songs_dir.is_dir():
            all_songs = [p for p in sorted(songs_dir.iterdir()) if p.is_file()]

            # Randomly sample up to max_num_songs from the available tracks
            if max_num_songs is not None:
                try:
                    n = int(max_num_songs)
                except (TypeError, ValueError):
                    n = None
            else:
                n = None

            if n is not None and n > 0 and n < len(all_songs):
                selected_songs = random.sample(all_songs, n)
            else:
                selected_songs = all_songs

            for p in selected_songs:
                modes.append(("song", {"path": p, "max_audio_len_s": max_audio_len_s}))

    return modes


def run_control_experiment(sim_cfg: Dict[str, Any], input_spec: Tuple[str, Dict[str, Any]]) -> None:
    """Placeholder for the actual control experiment from experiment_03.

    This will eventually:
      - Load / generate the input signal (white noise or song)
      - Configure the simulation according to sim_cfg
      - Run the adaptive controller and log/save results
    """
    mode, info = input_spec
    print("\n=== Running control experiment ===")
    print(f"Mode: {mode}")
    print(f"Input info: {info}")

    # TODO: plug in experiment_03 logic here

    # Load config variables that are accessed many times
    ROI = sim_cfg["ROI"]
    n_rirs = sim_cfg["n_rirs"]

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Acoustic path from actuator (speaker) to sensor (microphone)
    rirs, rirs_srs = load_rirs(root/'data'/'rir', max_n=n_rirs)
    rir_init = rirs[0]
    sr = rirs_srs[0]
    
    # Precompute RIR tensors and transition times for time-varying scenarios
    if n_rirs > 1:
        rirs_tensors = [torch.from_numpy(rir).float().to(device) for rir in rirs]
        max_rir_len = max(rir.shape[0] for rir in rirs_tensors)
        rirs_tensors = [F.pad(rir, (0, max_rir_len - rir.shape[0])) for rir in rirs_tensors]
        
        # Compute transition start/end times (in seconds)
        segment_duration_s = input_spec[1]["max_audio_len_s"] / n_rirs
        transition_times_s = [] 
        for i in range(1, n_rirs):
            transition_start_s = i * segment_duration_s
            assert (i==1) or (transition_start_s >= transition_times_s[-1][-1])
            transition_end_s = transition_start_s + sim_cfg["transition_time_s"]
            transition_times_s.append((transition_start_s, min(transition_end_s, input_spec[1]["max_audio_len_s"])))

    # Compute target response
    lem_delay = get_delay_from_ir(rir_init, sr)
    EQ_comp_dict = get_compensation_EQ_params(rir_init, sr, ROI, num_sections=6)
    target_mag_resp = EQ_comp_dict["target_response_db"]
    target_mag_freqs = EQ_comp_dict["freq_axis_smoothed"]

    # Initialize the LEM estimate (assume LEM is well-identified)
    LEM = torch.from_numpy(rir_init).view(1,1,-1).to(device)

    # Initialize differentiable EQ
    EQ = ParametricEQ(sample_rate=sr)
    dasp_param_dict = { k: torch.as_tensor(v, dtype=torch.float32).view(1) for k, v in EQ_comp_dict["eq_params"].items() }
    _, init_params_tensor = EQ.clip_normalize_param_dict(dasp_param_dict) # initial normalized parameter vector
    EQ_params = torch.nn.Parameter(init_params_tensor.clone().to(device))
    EQ_memory = 128+80 # TODO: hardcoded for now (should be greater than 0)
