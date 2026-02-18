from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch

from utils_ex04 import (
    load_config,
    iter_param_grid,
    discover_input_signals,
    run_control_experiment,
)

root = Path(__file__).resolve().parent.parent

def main() -> None:

    # Load configuration
    config_path = root / "configs" / "experiment_04_config.json"
    cfg = load_config(config_path)

    sim_param_grid = cfg.get("simulation_params", {})
    input_cfg = cfg.get("input", {})

    if not sim_param_grid:
        raise ValueError("No 'simulation_params' section found in config.")

    input_signals = discover_input_signals(input_cfg)
    if not input_signals:
        # Fallback: at least one white-noise configuration
        input_signals = [("white_noise", {"max_audio_len_s": None})]

    # Prepare aggregation structures for plotting
    # Unique transition times from the simulation grid
    tt_values = sim_param_grid.get("transition_time_s", [])
    unique_tt = sorted(set(tt_values))
    tt_to_idx = {tt: i for i, tt in enumerate(unique_tt)}

    # Aggregate results per (transition_time_s, optim_type)
    curves = defaultdict(list)  # key: (tt, optim_type) -> list of (time_axis, val_hist)
    tt_transitions = {}         # key: tt -> transition_times list (seconds)

    # Enumerate all combinations of simulation parameters
    for combo_idx, sim_cfg in enumerate(iter_param_grid(sim_param_grid)):
        print("\n############################################")
        print(f"Combination {combo_idx + 1}")
        print("Simulation config:")
        for k, v in sorted(sim_cfg.items()):
            print(f"  {k}: {v}")
        print("############################################")

        # For each parameter combo, run over all selected input signals
        results_per_probe = []
        for input_spec in input_signals:
            result = run_control_experiment(sim_cfg, input_spec)
            if result is None:
                continue
            results_per_probe.append(result)

            tt = result.get("transition_time_s", sim_cfg.get("transition_time_s"))
            optim = result.get("optim_type", sim_cfg.get("optim_type"))
            time_axis = np.asarray(result["time_axis"], dtype=float)
            val_hist = np.asarray(result["validation_error_history"], dtype=float)

            curves[(tt, optim)].append((time_axis, val_hist))

            # Cache transition start/end times per transition_time_s (in seconds)
            if tt not in tt_transitions:
                tt_transitions[tt] = result.get("transition_times", None)

        # end for input_spec

    # After all runs, build the summary figure
    if unique_tt:
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

        pass


if __name__ == "__main__":
    torch.manual_seed(123)
    main()
