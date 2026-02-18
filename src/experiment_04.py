import json
import itertools
import random
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple

import torch


root = Path(__file__).resolve().parent.parent


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
    print("Simulation config:")
    for k, v in sorted(sim_cfg.items()):
        print(f"  {k}: {v}")
    # TODO: plug in experiment_03 logic here


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
        print(f"Combination {combo_idx + 1}: {sim_cfg}")
        print("############################################")

        # For each parameter combo, run over all selected input signals
        for input_spec in input_signals:
            run_control_experiment(sim_cfg, input_spec)


if __name__ == "__main__":
    torch.manual_seed(123)
    main()
