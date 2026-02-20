import json
import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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
    curves = plot1_data["curves"]
    tt_transitions = plot1_data.get("tt_transitions", {})
    input_signals = plot1_data.get("input_signals", None)

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
