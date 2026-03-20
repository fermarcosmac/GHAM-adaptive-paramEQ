from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent.parent


def load_results(experiment_name: str) -> tuple[dict, dict, Path]:
    results_root = root / "results" / experiment_name
    cfg_path = results_root / "config.json"
    pkl_path = results_root / "plot1_data.pkl"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config file: {cfg_path}")
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing plot data file: {pkl_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    with pkl_path.open("rb") as f:
        plot_data = pickle.load(f)
    return cfg, plot_data, results_root


def _plot_mean_std(ax, series, color, label):
    if not series:
        return
    min_len = min(len(vals) for _, vals in series)
    t = None
    stack = []
    for ta, vals in series:
        ta = np.asarray(ta, dtype=float)
        vals = np.asarray(vals, dtype=float)
        if vals.size < min_len or np.isnan(vals).any():
            continue
        stack.append(vals[:min_len])
        if t is None:
            t = ta[:min_len]
    if not stack or t is None:
        return

    y = np.stack(stack, axis=0)
    avg = np.mean(y, axis=0)
    std = np.std(y, axis=0)

    ax.plot(t, avg, color=color, linewidth=1.1, label=label)
    n_mark = min(10, len(t))
    idx = np.linspace(0, len(t) - 1, num=n_mark, dtype=int)
    ax.errorbar(t[idx], avg[idx], yerr=std[idx], fmt="none", ecolor=color, elinewidth=0.9, capsize=3, alpha=0.75)


def plot_results(experiment_name: str) -> None:
    cfg, data, results_root = load_results(experiment_name)

    td_mse_curves = data.get("td_mse_curves", {})
    validation_curves = data.get("validation_curves", {})
    compute_time_stats = data.get("compute_time_stats", {})
    tt_transitions = data.get("tt_transitions", {})
    target_example = data.get("target_response_example", None)

    if not td_mse_curves or not validation_curves:
        print("No curves found in plot data.")
        return

    all_keys = sorted(set(td_mse_curves.keys()) | set(validation_curves.keys()))
    transition_times = sorted({k[0] for k in all_keys})
    algorithms = sorted({k[1] for k in all_keys})

    # Console compute-time table
    print("Average compute time per frame [s/frame]:")
    header = "transition_time_s" + "".join(f"\t{a}" for a in algorithms)
    print(header)
    ct_table = np.full((len(transition_times), len(algorithms)), np.nan)
    for i, tt in enumerate(transition_times):
        row = [f"{tt}"]
        for j, algo in enumerate(algorithms):
            stats = compute_time_stats.get((tt, algo), None)
            if stats and int(stats.get("total_frames", 0)) > 0:
                v = float(stats["total_time_s"]) / float(stats["total_frames"])
                ct_table[i, j] = v
                row.append(f"{v:.6f}")
            else:
                row.append("nan")
        print("\t".join(row))

    colors = plt.get_cmap("tab10")
    algo_color = {a: colors(i % 10) for i, a in enumerate(algorithms)}

    n_rows = len(transition_times)
    fig, axes = plt.subplots(n_rows, 2, figsize=(10, max(2.6, 2.5 * n_rows)), squeeze=False)

    for row, tt in enumerate(transition_times):
        ax_td = axes[row, 0]
        ax_val = axes[row, 1]

        for algo in algorithms:
            key = (tt, algo)
            _plot_mean_std(ax_td, td_mse_curves.get(key, []), algo_color[algo], algo)
            _plot_mean_std(ax_val, validation_curves.get(key, []), algo_color[algo], algo)

        for ax in (ax_td, ax_val):
            trans = tt_transitions.get(tt, None)
            if trans is not None:
                for t_start, t_end in trans:
                    ax.axvline(float(t_start), color="0.2", linestyle="--", linewidth=1.0, alpha=0.8)
                    if t_end != t_start:
                        ax.axvline(float(t_end), color="0.2", linestyle="--", linewidth=1.0, alpha=0.8)
                        ax.axvspan(float(t_start), float(t_end), color="0.85", alpha=0.35)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.8)
            if row == n_rows - 1:
                ax.set_xlabel("Time [s]")

        ax_td.set_yscale("log")
        ax_td.set_ylabel(f"TD-MSE\n(tt={tt}s)")
        ax_td.set_title("Time-domain MSE")

        ax_val.axhline(1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax_val.set_ylabel("Validation error")
        ax_val.set_title("Frequency-domain validation")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right", fontsize=8)

    # Optional target response inset (for quick reference)
    if target_example is not None and len(target_example.get("freq_axis", [])):
        ins = axes[0, 1].inset_axes([0.58, 0.57, 0.38, 0.38])
        f = np.asarray(target_example["freq_axis"], dtype=float)
        tdb = np.asarray(target_example["target_mag_db"], dtype=float)
        m = f > 0
        ins.plot(f[m], tdb[m], color="black", linewidth=1.0)
        ins.set_xscale("log")
        ins.set_title("Target H(f)", fontsize=7)
        ins.tick_params(axis="both", labelsize=6)
        ins.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    out_png = results_root / f"{experiment_name}_curves.png"
    fig.savefig(out_png, dpi=180)
    print(f"Saved figure: {out_png}")

    # Timing table figure
    fig_t, ax_t = plt.subplots(figsize=(max(6.0, 1.2 * len(algorithms) + 2.0), max(2.4, 0.6 * len(transition_times) + 1.8)))
    ax_t.axis("off")
    cell_text = [[f"{v:.6f}" if np.isfinite(v) else "-" for v in row] for row in ct_table]
    row_labels = [f"tt={tt}s" for tt in transition_times]
    tbl = ax_t.table(cellText=cell_text, rowLabels=row_labels, colLabels=algorithms, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    ax_t.set_title("Average per-frame compute time [s/frame]")
    fig_t.tight_layout()
    out_table = results_root / f"{experiment_name}_compute_time_table.png"
    fig_t.savefig(out_table, dpi=180)
    print(f"Saved timing table: {out_table}")

    plt.show()


def main() -> None:
    experiment_name = "experiment_05_ablation_debug"
    plot_results(experiment_name)


if __name__ == "__main__":
    main()
