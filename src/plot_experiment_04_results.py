import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from utils_ex04 import compute_parametric_eq_response

# Use LaTeX-style mathtext for figure text and numbers
mpl.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
    }
)

def load_results(experiment_name: str, root: Path) -> tuple[dict, dict]:
    results_root = root / "results" / experiment_name
    config_path = results_root / "config.json"
    plot_data_path = results_root / "plot1_data.pkl"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not plot_data_path.exists():
        raise FileNotFoundError(f"Plot data file not found: {plot_data_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    with plot_data_path.open("rb") as f:
        plot1_data = pickle.load(f)

    return cfg, plot1_data


def plot_results(cfg: dict, plot1_data: dict) -> None:
    curves = plot1_data["curves"]              # validation error curves
    loss_curves = plot1_data.get("loss_curves", {})  # training loss curves (optional)
    tt_transitions = plot1_data.get("tt_transitions", {})
    input_signals = plot1_data.get("input_signals", None)
    checkpoint_examples = plot1_data.get("checkpoint_examples", {})

    # curves keys are (transition_time_s, optim_type)
    unique_tt = sorted({tt for (tt, _) in curves.keys()})
    if not unique_tt:
        print("No curves found in plot data.")
        return

    # Pretty-print configuration in a similar style to experiment_04 logs
    print("Loaded configuration:\n")
    for top_key in sorted(cfg.keys()):
        section = cfg[top_key]
        print(f"{top_key}:")
        if isinstance(section, dict):
            for k, v in sorted(section.items()):
                print(f"  {k}: {v}")
        else:
            print(f"  {section}")
    print()

    if input_signals:
        print("Input signals used:")
        for sig in input_signals:
            print(f"  {sig}")
        print()

    fig, axes = plt.subplots(len(unique_tt), 1, figsize=(8, 3 * len(unique_tt)), sharex=False)
    if len(unique_tt) == 1:
        axes = [axes]

    # Assign a color per optimizer
    optim_types = sorted({opt for (_, opt) in curves.keys()})
    color_map = plt.get_cmap("tab10")
    optim_to_color = {opt: color_map(i % 10) for i, opt in enumerate(optim_types)}

    for idx, tt in enumerate(unique_tt):
        ax = axes[idx]
        ax.set_title(rf"$\mathrm{{Transition\ times:}}\ {tt}\ \mathrm{{s}}$")
        ax.set_ylabel(r"$\mathrm{Validation\ error}$")

        # Reference line at y=1
        ax.axhline(y=1.0, color="dimgray", linestyle="--", linewidth=1.0, alpha=0.6)

        # Plot vertical lines and shaded regions for transitions (if available)
        transitions = tt_transitions.get(tt, None)
        if transitions is not None:
            for t_start, t_end in transitions:
                if t_start == t_end:
                    # Instantaneous transition: single dark gray dashed line
                    ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                else:
                    # Start/end of transition: two dark gray dashed lines
                    ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                    ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                    # Dim shading over the transition zone
                    ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

        # For each optimizer, plot only the average curve with periodic std bars
        for optim in optim_types:
            key = (tt, optim)
            if key not in curves:
                continue

            series = curves[key]
            color = optim_to_color[optim]

            # Stack all probe curves to compute mean and std over probes
            min_len = min(len(v[1]) for v in series)
            all_vals = []
            common_time = None
            for time_axis, val_hist in series:
                t = time_axis[:min_len]
                v = val_hist[:min_len]
                all_vals.append(v)
                if common_time is None:
                    common_time = t

            if all_vals and common_time is not None:
                vals_stack = np.stack(all_vals, axis=0)
                avg_vals = np.mean(vals_stack, axis=0)
                std_vals = np.std(vals_stack, axis=0)

                # Plot average curve (thinner line) with plain-text legend label
                optim_label = optim.replace("_", " ")
                ax.plot(
                    common_time,
                    avg_vals,
                    color=color,
                    alpha=0.95,
                    linewidth=1.0,
                    label=optim_label,
                )

                # Add std bars at a sp5arser subset of points along the time axis
                num_markers = min(10, len(common_time))
                idxs = np.linspace(0, len(common_time) - 1, num=num_markers, dtype=int)
                ax.errorbar(
                    common_time[idxs],
                    avg_vals[idxs],
                    yerr=std_vals[idxs],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=3,
                    alpha=0.7,
                )

        if idx == len(unique_tt) - 1:
            ax.set_xlabel(r"Time [s]")

    # Add legend only once, in the top axis
    if len(axes) > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            axes[0].legend(handles, labels, loc="upper right")

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Training loss vs. time (separate figure, same style as validation)
    # ------------------------------------------------------------------
    if loss_curves:
        unique_tt_loss = sorted({tt for (tt, _) in loss_curves.keys()})
        fig_loss, axes_loss = plt.subplots(len(unique_tt_loss), 1, figsize=(8, 3 * len(unique_tt_loss)), sharex=False)
        if len(unique_tt_loss) == 1:
            axes_loss = [axes_loss]

        for idx, tt in enumerate(unique_tt_loss):
            ax = axes_loss[idx]
            # Match LaTeX-style formatting used in validation-error plots
            ax.set_title(rf"$\mathrm{{Transition\ times:}}\ {tt}\ \mathrm{{s}}$ (Loss)")
            ax.set_ylabel(r"$\mathrm{Loss}$")

            # Plot vertical lines and shaded regions for transitions (if available)
            transitions = tt_transitions.get(tt, None)
            if transitions is not None:
                for t_start, t_end in transitions:
                    if t_start == t_end:
                        ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                    else:
                        ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                        ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                        ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

            # For each optimizer, plot only the average curve with periodic std bars
            for optim in optim_types:
                key = (tt, optim)
                if key not in loss_curves:
                    continue

                series = loss_curves[key]
                color = optim_to_color[optim]

                min_len = min(len(v[1]) for v in series)
                all_vals = []
                common_time = None
                for time_axis, loss_hist in series:
                    t = time_axis[:min_len]
                    v = loss_hist[:min_len]
                    all_vals.append(v)
                    if common_time is None:
                        common_time = t

                if all_vals and common_time is not None:
                    vals_stack = np.stack(all_vals, axis=0)
                    avg_vals = np.mean(vals_stack, axis=0)
                    std_vals = np.std(vals_stack, axis=0)

                    # For log-scale plotting, avoid zeros/negatives
                    avg_plot = np.clip(avg_vals, 1e-8, None)

                    optim_label = optim.replace("_", " ")
                    ax.plot(
                        common_time,
                        avg_plot,
                        color=color,
                        alpha=0.95,
                        linewidth=1.0,
                        label=optim_label,
                    )

                    num_markers = min(10, len(common_time))
                    idxs = np.linspace(0, len(common_time) - 1, num=num_markers, dtype=int)
                    ax.errorbar(
                        common_time[idxs],
                        avg_plot[idxs],
                        yerr=std_vals[idxs],
                        fmt="none",
                        ecolor=color,
                        elinewidth=1.0,
                        capsize=3,
                        alpha=0.7,
                    )

            # Use logarithmic y-axis for loss curves
            ax.set_yscale("log")

            if idx == len(unique_tt_loss) - 1:
                ax.set_xlabel(r"Time [s]")

        # Legend on top axis only
        if len(axes_loss) > 0:
            handles_l, labels_l = axes_loss[0].get_legend_handles_labels()
            if handles_l:
                axes_loss[0].legend(handles_l, labels_l, loc="upper right")

        plt.tight_layout()

    # ------------------------------------------------------------------
    # Checkpoint-based response tiles: one example run per optimizer
    # ------------------------------------------------------------------
    if checkpoint_examples:
        optim_keys = sorted(checkpoint_examples.keys())
        # Number of checkpoints per optimizer (assume consistent within each list)
        n_rows = len(optim_keys)
        n_cols = max(len(checkpoint_examples[opt]) for opt in optim_keys)

        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
        if n_rows == 1:
            axes2 = np.array([axes2])
        if n_cols == 1:
            axes2 = axes2[:, np.newaxis]

        for row, opt in enumerate(optim_keys):
            checkpoints = checkpoint_examples[opt]
            color_total = "tab:blue"
            color_lem = "tab:green"
            color_desired = "tab:orange"
            color_eq = "tab:red"

            for col, cp in enumerate(checkpoints):
                if col >= n_cols:
                    break
                ax = axes2[row, col]

                freqs_log = np.asarray(cp["freqs_log"], dtype=float)
                H_total = np.asarray(cp["H_total_db"], dtype=float)
                H_desired = np.asarray(cp["H_desired_db"], dtype=float)
                H_lem = np.asarray(cp["H_lem_db"], dtype=float)

                # Prefer denormalized EQ_matrix stored by the experiment. If not
                # available (e.g., older results), fall back to reshaping the
                # flattened normalized EQ_params.
                eq_matrix = cp.get("EQ_matrix", None)
                if eq_matrix is None:
                    eq_params_flat = cp.get("EQ_params", None)
                    if eq_params_flat is not None:
                        eq_params_flat = np.asarray(eq_params_flat, dtype=float).ravel()
                        if eq_params_flat.size % 3 == 0:
                            n_filters = eq_params_flat.size // 3
                            eq_matrix = eq_params_flat.reshape(n_filters, 3)
                sr_cp = float(cp.get("sr", cfg.get("sample_rate", 48000)))
                t_s = float(cp.get("time_s", 0.0))

                ax.plot(freqs_log, H_desired, color=color_desired, linewidth=1.0, alpha=0.9, label="Desired")
                ax.plot(freqs_log, H_lem, color=color_lem, linewidth=0.8, alpha=0.9, label="LEM")
                ax.plot(freqs_log, H_total, color=color_total, linewidth=1.0, alpha=0.9, label="EQ+LEM")

                # Overlay EQ-only frequency response if available
                if eq_matrix is not None:
                    eq_matrix_np = np.asarray(eq_matrix, dtype=float)
                    try:
                        H_eq_db = compute_parametric_eq_response(eq_matrix_np, freqs_log, sr_cp)
                        ax.plot(freqs_log, H_eq_db, color=color_eq, linewidth=0.8, alpha=0.9, label="EQ")
                    except Exception:
                        pass

                ax.set_xscale("log")
                if row == 0:
                    ax.set_title(f"t = {t_s:.1f} s")
                if row == n_rows - 1:
                    ax.set_xlabel("Frequency [Hz]")
                if col == 0:
                    ax.set_ylabel("Magnitude [dB]")

                if row == 0 and col == 0:
                    ax.legend(loc="best")

        # Add optimizer names as text on the left of each row
        for row, opt in enumerate(optim_keys):
            label = opt.replace("_", " ")
            axes2[row, 0].text(
                0.02,
                0.95,
                label,
                transform=axes2[row, 0].transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7),
            )

        plt.tight_layout()
    plt.show()


def main() -> None:
    # Select the experiment to plot here
    experiment_name = "experiment_04_run_01"

    # Project root (same convention as experiment_04.py)
    root = Path(__file__).resolve().parents[1]

    cfg, plot1_data = load_results(experiment_name, root)
    print(f"Experiment name: {experiment_name}")
    plot_results(cfg, plot1_data)


if __name__ == "__main__":
    main()
