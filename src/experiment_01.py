from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import numpy as np
from utils import (
    load_audio,
    save_audio,
    load_rirs,
    ensure_rirs_sample_rate,
    simulate_time_varying_rir,
    rms,
)

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



def octave_average(f: np.ndarray, resp: np.ndarray, bpo: int, 
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



def eq_optimizer(
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
    init_params, lb, ub = init_eq_parameters(num_filters, ROI)

    res = least_squares(
        eq_objective_function,                                          # loss
        init_params.ravel(),                                            # initial parameters (flattened)
        bounds=(lb.ravel(), ub.ravel()),
        args=(num_filters, f, meas_resp_db, target_resp_db, ROI, Fs),   # additional args to compute loss function
        ftol=1e-8,
        max_nfev=max(500 * num_filters, 3600),
        verbose=2,
    )

    EQ_params = res.x.reshape(num_filters, 3)                           # Extract optimized (LS) EQ parameters
    filtResp = compute_parametric_eq_response(EQ_params, f, Fs)
    outputResp = meas_resp_db + filtResp

    return EQ_params, outputResp, filtResp


def eq_objective_function(
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




def init_eq_parameters(
    num_filters: int,
    ROI: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Initialize EQ parameter matrix x0 and bounds lb / ub.

    Parameter format per filter:
        [gain_dB, Q_or_slope, Fc]
    """
    fmin = max(20.0, ROI[0])
    fmax = min(20e3, ROI[1])

    # Center frequencies
    Fc_peaks = np.logspace(         # All but last two are peaking filters
        np.log10(fmin),
        np.log10(fmax),
        num_filters - 2
    )
    Fc_shelves = np.array([         # Last two filters are shelves (high and low)
        (2 * ROI[0] + ROI[1]) / 3,
        (ROI[0] + 2 * ROI[1]) / 3
    ])

    # Initial parameter matrix
    init_params = np.zeros((num_filters, 3))
    init_params[:, 0] = get_initial_gains(num_filters)
    init_params[:-2, 1] = 1.5     # Q for peak filters
    init_params[-2:, 1] = 1.0     # slope for shelf filters
    init_params[:-2, 2] = Fc_peaks
    init_params[-2:, 2] = Fc_shelves

    # Lower bounds
    lb = np.zeros_like(init_params)
    lb[:, 0] = -24.0        # gain lower bound
    lb[:-2, 1] = 0.2        # Q for peak filters lower bound
    lb[-2:, 1] = 0.1        # slope for shelf filters lower bound
    lb[:, 2] = fmin-1e-6    # Fc lower bound

    # Upper bounds
    ub = np.zeros_like(init_params)
    ub[:, 0] = 20.0         # gain upper bound
    ub[:-2, 1] = 17.3       # Q for peak filters upper bound
    ub[-2:, 1] = 5.0        # slope for shelf filters upper bound
    ub[:, 2] = fmax+1e-6    # Fc upper bound

    return init_params, lb, ub



def get_initial_gains(n: int, seed: int = 0) -> np.ndarray:
    """
    Generate initial gain values for EQ filters.
    Uses a fixed RNG seed so results are repeatable.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)

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





def get_compensation_EQ_params(rir: np.ndarray, sr: int, ROI: Tuple[float, float]=(20.0, 20000.0)) -> dict:
    """Estimate Parametric EQ parameters to compensate for the given RIR.

    This function analyzes the RIR
    and compute suitable EQ parameters to flatten its frequency response.
    """
    nfft = len(rir)
    freq_response = np.abs(np.fft.rfft(rir, n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1/sr)

    # Apply octave averaging (no smoothing for filter responses)
    oa, cf = octave_average(freqs, freq_response, bpo=24, freq_range=ROI, b_smooth=False)

    # Compute target response for compensation EQ
    target_resp, pfit, pdb = _get_target_response_comp_EQ(cf, oa, ROI)

    # Optimize parametric EQ to match target response
    EQ_params, out_resp_db, filt_resp_db = eq_optimizer(
    num_filters=6,                                      # 6 filters (pytorch implementation limitation)
    f=cf,
    meas_resp_db=pdb,
    target_resp_db=target_resp,
    ROI=ROI,
    Fs=sr,
)


    plt.figure()
    plt.semilogx(freqs, 20*np.log10(freq_response + 1e-12), label="RIR Frequency Response")
    plt.semilogx(cf, 20*np.log10(oa + 1e-12), label="Octave-Averaged Response")
    plt.semilogx(cf, target_resp, label="Target Compensation Response", linestyle='--')
    plt.semilogx(cf, out_resp_db, label="Equalized Response", linestyle='-.')
    plt.semilogx(cf, filt_resp_db, label="EQ Filter Response", linestyle=':')
    plt.axvline(ROI[0], color="red", linestyle="--", label="ROI Limits" if ROI else "")
    plt.axvline(ROI[1], color="red", linestyle="--" if ROI else "")
    plt.legend(), plt.xlabel("Frequency (Hz)"), plt.ylabel("Magnitude (dB)")
    plt.title("RIR Frequency Response and Target Compensation EQ")
    plt.savefig("rir_response.png",dpi=150, bbox_inches='tight')

    # build parameter dict to return as output... TODO
    # Think which responses to return... TODO





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

    # Initial room impulse response measurements
    rir_init = rirs[0]
    EQ = get_compensation_EQ_params(rir_init, sr, ROI)

    # Parametric EQ estimation for virtual room compensation

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