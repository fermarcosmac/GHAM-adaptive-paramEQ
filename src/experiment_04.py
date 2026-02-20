from pathlib import Path
from collections import defaultdict
import json
import pickle
import random

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


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU/CUDA) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:

    # Load configuration
    config_path = root / "configs" / "experiment_04_config.json"
    cfg = load_config(config_path)

    # Global seeding for reproducibility (EQ init, white noise, song sampling)
    seed = int(cfg.get("seed", 123))

    experiment_name = cfg.get("experiment_name", "experiment_04")
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
    input_ids_used = set()      # input identifiers (paths or labels) actually used
    checkpoint_examples = {}    # optim_type -> list of checkpoint states from a representative run

    # Pair optim_type and mu_opt
    optim_list = sim_param_grid.get("optim_type", [])
    mu_list = sim_param_grid.get("mu_opt", [])

    if optim_list and mu_list and len(optim_list) != len(mu_list):
        raise ValueError(
            "Length mismatch between 'optim_type' and 'mu_opt' in config: "
            f"got {len(optim_list)} optimizers and {len(mu_list)} mu_opt values. "
            "They must have the same length so each optimizer pairs with one mu_opt."
        )

    # All other parameters still form a full grid
    base_param_grid = {
        k: v for k, v in sim_param_grid.items() if k not in ("optim_type", "mu_opt")
    }

    # Pre-compute total number of parameter combinations for logging
    base_cfgs = list(iter_param_grid(base_param_grid))
    num_opt_mu = len(optim_list) if (optim_list and mu_list) else 0
    total_combos = len(base_cfgs) * max(1, num_opt_mu)

    combo_idx = 0
    for base_cfg in base_cfgs:

        if optim_list and mu_list:
            opt_mu_pairs = zip(optim_list, mu_list)

        for optim, mu in opt_mu_pairs:
            sim_cfg = dict(base_cfg)
            if optim is not None:
                sim_cfg["optim_type"] = optim
            if mu is not None:
                sim_cfg["mu_opt"] = mu

            combo_idx += 1
            print("\n############################################")
            if total_combos > 0:
                print(f"Combination {combo_idx}/{total_combos}")
            else:
                print(f"Combination {combo_idx}")
            print("Simulation config:")
            for k, v in sorted(sim_cfg.items()):
                print(f"  {k}: {v}")
            print("############################################")

            # For each parameter combo, run over all selected input signals
            results_per_probe = []
            for input_spec in input_signals:
                set_seed(seed) # Same initial params and noisy input
                result = run_control_experiment(sim_cfg, input_spec)
                if result is None:
                    continue

                # Track which input signals were used (audio file paths or white_noise label)
                input_id = None
                if isinstance(input_spec, (list, tuple)) and len(input_spec) == 2:
                    mode, info = input_spec
                    if mode == "white_noise":
                        input_id = "white_noise"
                    elif isinstance(info, dict) and "path" in info:
                        input_id = str(info["path"])
                    else:
                        input_id = str(mode)
                else:
                    input_id = str(input_spec)
                input_ids_used.add(str(input_id))
                results_per_probe.append(result)

                tt = result.get("transition_time_s", sim_cfg.get("transition_time_s"))
                optim_used = result.get("optim_type", sim_cfg.get("optim_type"))
                time_axis = np.asarray(result["time_axis"], dtype=float)
                val_hist = np.asarray(result["validation_error_history"], dtype=float)

                curves[(tt, optim_used)].append((time_axis, val_hist))

                # Cache transition start/end times per transition_time_s (in seconds)
                if tt not in tt_transitions:
                    tt_transitions[tt] = result.get("transition_times", None)

                # Store one set of checkpoint examples per optimizer (first occurrence)
                if "checkpoints" in result and optim_used not in checkpoint_examples:
                    checkpoint_examples[optim_used] = result["checkpoints"]
            # end for input_spec

    # After all runs, serialize configuration and plotting data to results folder
    results_root = root / "results" / experiment_name
    results_root.mkdir(parents=True, exist_ok=True)

    # Save a copy of the configuration used for this experiment
    config_out_path = results_root / "config.json"
    with config_out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    # Save plotting data (curves, transition times, input signals, and checkpoint examples) for later visualization
    plot1_data = {
        "curves": curves,
        "tt_transitions": tt_transitions,
        "input_signals": sorted(input_ids_used),
        "checkpoint_examples": checkpoint_examples,
    }
    plot_out_path = results_root / "plot1_data.pkl"
    with plot_out_path.open("wb") as f:
        pickle.dump(plot1_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved config to: {config_out_path}")
    print(f"Saved plotting data to: {plot_out_path}")
if __name__ == "__main__":
    main()
