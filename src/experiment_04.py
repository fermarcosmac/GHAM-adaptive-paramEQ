from pathlib import Path
import torch
from utils_ex04 import (
    load_config,
    iter_param_grid,
    discover_input_signals,
    run_control_experiment
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

    # Enumerate all combinations of simulation parameters
    for combo_idx, sim_cfg in enumerate(iter_param_grid(sim_param_grid)):
        print("\n############################################")
        print(f"Combination {combo_idx + 1}")
        print("Simulation config:")
        for k, v in sorted(sim_cfg.items()):
            print(f"  {k}: {v}")
        print("############################################")

        # For each parameter combo, run over all selected input signals
        for input_spec in input_signals:
            run_control_experiment(sim_cfg, input_spec)


if __name__ == "__main__":
    torch.manual_seed(123)
    main()
