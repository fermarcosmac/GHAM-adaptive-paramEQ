from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from scipy.signal import lfilter, minimum_phase

from lib.local_pyaec.frequency_domain_adaptive_filters import fxfdaf as lib_fxfdaf
from utils_ex04 import (
    build_target_response_lin_phase,
    discover_input_signals,
    get_compensation_EQ_params,
    get_delay_from_ir,
    interp_to_log_freq,
    iter_param_grid,
    load_config,
    load_rirs,
    run_control_experiment,
    update_LEM,
)

root = Path(__file__).resolve().parent.parent


def prepare_rir_context(sim_cfg: Dict[str, Any], max_audio_len_s: float, device: torch.device) -> Dict[str, Any]:
    """Load and prepare RIR tensors and transition schedule using experiment_04 logic."""
    n_rirs = int(sim_cfg["n_rirs"])
    rir_dir = Path(sim_cfg["rir_dir"])
    rirs, rirs_srs = load_rirs(rir_dir, max_n=n_rirs, normalize=False)
    if not rirs:
        raise ValueError(f"No RIRs found in: {rir_dir}")

    sr = int(rirs_srs[0])
    rirs_tensors = [torch.from_numpy(r).float().to(device) for r in rirs]
    max_rir_len = max(r.shape[-1] for r in rirs_tensors)
    rirs_tensors = [F.pad(r, (0, max_rir_len - r.shape[-1])) for r in rirs_tensors]

    transition_times_s = []
    if n_rirs > 1:
        segment_duration_s = max_audio_len_s / n_rirs
        tt = float(sim_cfg["transition_time_s"])
        for i in range(1, n_rirs):
            t_start = i * segment_duration_s
            t_end = min(t_start + tt, max_audio_len_s)
            transition_times_s.append((float(t_start), float(t_end)))

    return {
        "sr": sr,
        "rirs": rirs,
        "rirs_tensors": rirs_tensors,
        "transition_times_s": transition_times_s,
    }


def load_input_for_sr(input_spec: Tuple[str, Dict[str, Any]], sr: int, device: torch.device) -> np.ndarray:
    """Load one input signal as mono float32 np array at requested sample rate."""
    mode, info = input_spec
    max_audio_len_s = info.get("max_audio_len_s", None)

    if mode == "white_noise":
        if max_audio_len_s is None:
            raise ValueError("white_noise requires max_audio_len_s in input spec")
        n = int(float(max_audio_len_s) * sr)
        x = torch.randn(n, device=device)
    else:
        audio_path = Path(info["path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        wav, wav_sr = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(wav_sr) != sr:
            wav = torchaudio.transforms.Resample(orig_freq=wav_sr, new_freq=sr)(wav)
        wav = wav.squeeze(0).to(device)

        if max_audio_len_s is not None:
            max_samples = int(float(max_audio_len_s) * sr)
            wav = wav[:max_samples]
        x = wav

    peak = torch.max(torch.abs(x)).item()
    if peak > 0:
        x = x / peak
    return x.detach().cpu().numpy().astype(np.float64)


def build_target_response_np(rir_init: np.ndarray, sr: int, sim_cfg: Dict[str, Any], device: torch.device) -> Dict[str, np.ndarray]:
    """Build target impulse response and target magnitude response using experiment_04 utilities."""
    roi = tuple(sim_cfg["ROI"])
    target_response_type = sim_cfg["target_response_type"]

    lem_delay = get_delay_from_ir(rir_init, sr)
    eq_comp = get_compensation_EQ_params(rir_init, sr, ROI=roi, num_sections=7)
    target_mag_resp = eq_comp["target_response_db"]
    target_mag_freqs = eq_comp["freq_axis_smoothed"]

    h_target = build_target_response_lin_phase(
        sr=sr,
        response_type=target_response_type,
        target_mag_resp=target_mag_resp,
        target_mag_freqs=target_mag_freqs,
        fir_len=2048,
        ROI=roi,
        rolloff_octaves=0.5,
        device=device,
    )

    h_linear_np = h_target.squeeze().detach().cpu().numpy()
    h_min_np = minimum_phase(h_linear_np, method="homomorphic", half=False)
    h_target_np = np.concatenate([np.zeros(int(lem_delay), dtype=np.float64), h_min_np.astype(np.float64)]) / 2.0

    nfft = int(2 * int(sim_cfg["frame_len"]) - 1)
    f = np.fft.rfftfreq(nfft, d=1.0 / sr)
    target_db = 20.0 * np.log10(np.abs(np.fft.rfft(h_target_np, n=nfft)) + 1e-8)

    return {
        "h_target": h_target_np,
        "freqs": f,
        "target_db": target_db,
    }


def _smooth_log_mag_numpy(mag_db: np.ndarray, freqs: np.ndarray, roi: Tuple[float, float], n_points: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate to log frequency and apply moving-average smoothing."""
    mask = (freqs >= roi[0]) & (freqs <= roi[1])
    f_roi = torch.from_numpy(freqs[mask].astype(np.float32))
    mag_roi = torch.from_numpy(mag_db[mask].astype(np.float32))

    mag_log, f_log = interp_to_log_freq(mag_roi, f_roi, n_points=n_points)

    k = 15
    pad = k // 2
    ker = torch.ones(1, 1, k, dtype=torch.float32) / float(k)
    x = mag_log.view(1, 1, -1)
    x = F.pad(x, (pad, pad), mode="reflect")
    smoothed = F.conv1d(x, ker, padding=0).squeeze(0).squeeze(0)

    return f_log.cpu().numpy().astype(np.float64), smoothed.cpu().numpy().astype(np.float64)


def _compute_validation_error(
    w_ctrl: np.ndarray,
    lem_ir: np.ndarray,
    target_db: np.ndarray,
    freqs: np.ndarray,
    roi: Tuple[float, float],
    prev_total_db: np.ndarray | None,
    forget_factor: float,
) -> Tuple[float, np.ndarray]:
    """Compute smoothed frequency-domain validation error following experiment_04 style."""
    nfft = int((len(freqs) - 1) * 2)
    h_ctrl = np.fft.rfft(w_ctrl, n=nfft)
    h_lem = np.fft.rfft(lem_ir, n=nfft)
    h_total = h_ctrl * h_lem

    total_db_current = 20.0 * np.log10(np.abs(h_total) + 1e-8)
    if prev_total_db is None:
        total_db = total_db_current
    else:
        total_db = float(forget_factor) * total_db_current + (1.0 - float(forget_factor)) * prev_total_db

    lem_db = 20.0 * np.log10(np.abs(h_lem) + 1e-8)

    _, total_s = _smooth_log_mag_numpy(total_db, freqs, roi)
    _, target_s = _smooth_log_mag_numpy(target_db, freqs, roi)
    _, lem_s = _smooth_log_mag_numpy(lem_db, freqs, roi)

    num = float(np.mean(np.abs(total_s - target_s)))
    den = float(np.mean(np.abs(lem_s - target_s))) + 1e-12
    return num / den, total_db


def _fxlms_frame(
    x_block: np.ndarray,
    d_block: np.ndarray,
    lem_ir: np.ndarray,
    h_hat: np.ndarray,
    w: np.ndarray,
    mu: float,
    u_state: np.ndarray,
    u_f_state: np.ndarray,
    x_hat_state: np.ndarray,
    sec_state: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One frame update for sample-wise FxLMS with explicit secondary-path simulation."""
    m = len(x_block)
    e = np.zeros(m, dtype=np.float64)
    y_ctrl = np.zeros(m, dtype=np.float64)
    y_sec = np.zeros(m, dtype=np.float64)

    for i in range(m):
        u_state[1:] = u_state[:-1]
        u_state[0] = x_block[i]
        y_ctrl_i = float(np.dot(w, u_state))

        sec_state[1:] = sec_state[:-1]
        sec_state[0] = y_ctrl_i
        y_sec_i = float(np.dot(lem_ir, sec_state))

        e_i = d_block[i] - y_sec_i

        x_hat_state[1:] = x_hat_state[:-1]
        x_hat_state[0] = x_block[i]
        x_f_i = float(np.dot(h_hat, x_hat_state))

        u_f_state[1:] = u_f_state[:-1]
        u_f_state[0] = x_f_i
        w += mu * e_i * u_f_state

        e[i] = e_i
        y_ctrl[i] = y_ctrl_i
        y_sec[i] = y_sec_i

    return e, y_ctrl, y_sec, w, u_state, u_f_state, x_hat_state


class _LocalFxFDAF:
    """Simple block FxFDAF fallback when library fxfdaf is not implemented."""

    def __init__(self, block_size: int, mu: float, beta: float):
        self.m = int(block_size)
        self.mu = float(mu)
        self.beta = float(beta)
        self.window = np.hanning(self.m)
        self.h = np.zeros(self.m + 1, dtype=np.complex128)
        self.norm = np.full(self.m + 1, 1e-8, dtype=np.float64)
        self.x_old = np.zeros(self.m, dtype=np.float64)
        self.xf_old = np.zeros(self.m, dtype=np.float64)

        self.ctrl_state = np.zeros(self.m, dtype=np.float64)
        self.sec_state = None
        self.xhat_state = None

    def run_frame(self, x: np.ndarray, d: np.ndarray, lem_ir: np.ndarray, h_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.sec_state is None:
            self.sec_state = np.zeros(len(lem_ir), dtype=np.float64)
        if self.xhat_state is None:
            self.xhat_state = np.zeros(len(h_hat), dtype=np.float64)

        # Build filtered-x block for adaptation.
        xf = np.zeros_like(x)
        for i in range(len(x)):
            self.xhat_state[1:] = self.xhat_state[:-1]
            self.xhat_state[0] = x[i]
            xf[i] = float(np.dot(h_hat, self.xhat_state))

        x2 = np.concatenate([self.x_old, x])
        xf2 = np.concatenate([self.xf_old, xf])
        self.x_old = x.copy()
        self.xf_old = xf.copy()

        x2_fft = np.fft.rfft(x2)
        xf2_fft = np.fft.rfft(xf2)

        y_ctrl = np.fft.irfft(self.h * x2_fft, n=2 * self.m)[self.m:]

        y_sec = np.zeros_like(y_ctrl)
        for i in range(len(y_ctrl)):
            self.sec_state[1:] = self.sec_state[:-1]
            self.sec_state[0] = y_ctrl[i]
            y_sec[i] = float(np.dot(lem_ir, self.sec_state))

        e = d - y_sec

        e2 = np.concatenate([np.zeros(self.m, dtype=np.float64), e * self.window])
        e2_fft = np.fft.rfft(e2)

        self.norm = self.beta * self.norm + (1.0 - self.beta) * (np.abs(xf2_fft) ** 2)
        g = self.mu * e2_fft / (self.norm + 1e-3)
        self.h = self.h + np.conj(xf2_fft) * g

        h_td = np.fft.irfft(self.h, n=2 * self.m)
        h_td[self.m:] = 0.0
        self.h = np.fft.rfft(h_td)

        return e, y_ctrl, y_sec, h_td[: self.m].copy()


def run_fir_baseline_experiment(
    sim_cfg: Dict[str, Any],
    input_spec: Tuple[str, Dict[str, Any]],
    algorithm: str,
    algo_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Run one FIR baseline (FxLMS/FxFDAF) and return per-frame curves."""
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame_len = int(sim_cfg["frame_len"])
    hop_len = int(sim_cfg["hop_len"])
    if frame_len != hop_len:
        raise ValueError("For current FIR baseline implementation, require frame_len == hop_len")

    rir_ctx = prepare_rir_context(sim_cfg, float(input_spec[1]["max_audio_len_s"]), device)
    sr = int(rir_ctx["sr"])

    x = load_input_for_sr(input_spec, sr=sr, device=device)
    if x.size < frame_len:
        raise ValueError("Input shorter than frame_len")

    target_ctx = build_target_response_np(rir_ctx["rirs"][0], sr, sim_cfg, device)
    h_target = target_ctx["h_target"]
    freqs = target_ctx["freqs"]
    target_db = target_ctx["target_db"]

    d_full = np.convolve(x, h_target, mode="full")[: len(x)]

    mu = float(algo_cfg.get("mu", 0.01))
    beta = float(algo_cfg.get("beta", 0.9))
    n_ctrl = int(algo_cfg.get("filter_len", frame_len))
    n_ctrl = max(8, min(n_ctrl, frame_len * 2))

    h_hat = np.asarray(rir_ctx["rirs"][0], dtype=np.float64)
    if h_hat.ndim != 1:
        h_hat = h_hat.reshape(-1)

    w = np.zeros(n_ctrl, dtype=np.float64)
    u_state = np.zeros(n_ctrl, dtype=np.float64)
    u_f_state = np.zeros(n_ctrl, dtype=np.float64)
    x_hat_state = np.zeros(len(h_hat), dtype=np.float64)
    sec_state = np.zeros(len(rir_ctx["rirs"][0]), dtype=np.float64)

    fdf = _LocalFxFDAF(block_size=frame_len, mu=mu, beta=beta)

    n_frames = (len(x) - frame_len) // hop_len + 1
    td_mse_history: List[float] = []
    val_history: List[float] = []
    prev_total_db = None

    y_control = np.zeros(len(x), dtype=np.float64)
    y_out = np.zeros(len(x), dtype=np.float64)

    roi = tuple(sim_cfg["ROI"])
    forget_factor = float(sim_cfg["forget_factor"])

    transition_times = rir_ctx["transition_times_s"] if int(sim_cfg["n_rirs"]) > 1 else None

    lib_fxfdaf_available = True
    if algorithm == "FxFDAF":
        try:
            _ = lib_fxfdaf.fxfdaf(np.zeros(frame_len), np.zeros(frame_len), h_hat, frame_len, mu=mu, beta=beta)
        except Exception:
            lib_fxfdaf_available = False

    for k in range(n_frames):
        start = k * hop_len
        stop = start + frame_len
        now_s = float(start / sr)

        lem_t = update_LEM(now_s, int(sim_cfg["n_rirs"]), transition_times or [], rir_ctx["rirs_tensors"])
        lem_ir = lem_t.squeeze().detach().cpu().numpy().astype(np.float64)

        x_fr = x[start:stop]
        d_fr = d_full[start:stop]

        if algorithm == "FxLMS":
            e_fr, y_ctrl_fr, y_out_fr, w, u_state, u_f_state, x_hat_state = _fxlms_frame(
                x_fr,
                d_fr,
                lem_ir,
                h_hat,
                w,
                mu,
                u_state,
                u_f_state,
                x_hat_state,
                sec_state,
            )
        elif algorithm == "FxFDAF":
            if lib_fxfdaf_available:
                # Library path is currently unimplemented in local_pyaec; keep fallback deterministic.
                e_fr, y_ctrl_fr, y_out_fr, w_fr = fdf.run_frame(x_fr, d_fr, lem_ir, h_hat)
            else:
                e_fr, y_ctrl_fr, y_out_fr, w_fr = fdf.run_frame(x_fr, d_fr, lem_ir, h_hat)
            w = np.zeros_like(w)
            w[: min(len(w), len(w_fr))] = w_fr[: min(len(w), len(w_fr))]
        else:
            raise ValueError(f"Unknown FIR algorithm: {algorithm}")

        y_control[start:stop] = y_ctrl_fr
        y_out[start:stop] = y_out_fr

        td_mse = float(np.mean(np.square(e_fr)))
        val_err, prev_total_db = _compute_validation_error(
            w_ctrl=w,
            lem_ir=lem_ir,
            target_db=target_db,
            freqs=freqs,
            roi=roi,
            prev_total_db=prev_total_db,
            forget_factor=forget_factor,
        )

        td_mse_history.append(td_mse)
        val_history.append(val_err)

    elapsed = float(time.perf_counter() - t0)
    time_axis = np.arange(n_frames, dtype=np.float64) * (hop_len / sr)

    return {
        "algorithm": algorithm,
        "time_axis": time_axis,
        "td_mse_history": np.asarray(td_mse_history, dtype=np.float64),
        "validation_error_history": np.asarray(val_history, dtype=np.float64),
        "transition_times": transition_times,
        "transition_time_s": float(sim_cfg["transition_time_s"]),
        "input_audio": x.astype(np.float32),
        "desired_audio": d_full.astype(np.float32),
        "y_control": y_control.astype(np.float32),
        "y_out": y_out.astype(np.float32),
        "target_freq_axis": freqs.astype(np.float32),
        "target_mag_db": target_db.astype(np.float32),
        "sr": sr,
        "n_frames": int(n_frames),
        "control_experiment_time_s": elapsed,
        "avg_compute_time_per_frame_s": elapsed / n_frames if n_frames > 0 else float("nan"),
    }


def build_proposed_sim_cfg(shared_cfg: Dict[str, Any], proposed_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Compose a run_control_experiment-compatible cfg from shared + entry config."""
    sim = dict(shared_cfg)
    sim.update(proposed_entry)

    sim.setdefault("loss_type", "FD-MSE")
    sim.setdefault("optim_type", "SGD")
    sim.setdefault("mu_opt", 0.005)
    sim.setdefault("lambda_newton", 1.0)
    sim.setdefault("eps_0", 0.0)
    sim.setdefault("n_checkpoints", 0)
    sim.setdefault("use_true_LEM", False)
    sim.setdefault("debug_plot", False)

    return sim
