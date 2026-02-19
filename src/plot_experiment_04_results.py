import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
        ax.set_title(f"transition_time_s = {tt}")
        ax.set_ylabel("Validation error")
        ax.grid(True, alpha=0.3)

        # Reference line at y=1
        ax.axhline(y=1.0, color="dimgray", linestyle="--", linewidth=1.0, alpha=0.6)

        # Plot vertical lines for transitions (if available)
        transitions = tt_transitions.get(tt, None)
        if transitions is not None:
            for t_start, t_end in transitions:
                if t_start == t_end:
                    ax.axvline(x=t_start, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
                else:
                    ax.axvline(x=t_start, color="green", linestyle="--", linewidth=1.0, alpha=0.8)
                    ax.axvline(x=t_end, color="red", linestyle="--", linewidth=1.0, alpha=0.8)

        # For each optimizer, overlay individual curves and their average
        for optim in optim_types:
            key = (tt, optim)
            if key not in curves:
                continue

            series = curves[key]
            color = optim_to_color[optim]

            # Plot individual probe curves (dimmed)
            min_len = min(len(v[1]) for v in series)
            all_vals = []
            common_time = None
            for time_axis, val_hist in series:
                t = time_axis[:min_len]
                v = val_hist[:min_len]
                ax.plot(t, v, color=color, alpha=0.2, linewidth=0.8)
                all_vals.append(v)
                if common_time is None:
                    common_time = t

            # Plot average curve (highlighted)
            if all_vals and common_time is not None:
                avg_vals = np.mean(np.stack(all_vals, axis=0), axis=0)
                ax.plot(common_time, avg_vals, color=color, alpha=0.95, linewidth=2.0, label=f"{optim}")

        if idx == len(unique_tt) - 1:
            ax.set_xlabel("Time (s)")

    # Add legend only once, in the last axis
    if len(axes) > 0:
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            axes[-1].legend(handles, labels, loc="upper right")

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
