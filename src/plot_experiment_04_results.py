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
    loss_curves = plot1_data.get("loss_curves", {})
    tt_transitions = plot1_data.get("tt_transitions", {})
    input_signals = plot1_data.get("input_signals", None)
    checkpoint_examples = plot1_data.get("checkpoint_examples", {})

    # curves keys are (transition_time_s, optim_type)
    # Derive dimensions from 3-tuple keys: (transition_time_s, optim_type, loss_type)
    all_curve_keys = list(curves.keys())
    if not all_curve_keys:
        print("No curves found in plot data.")
        return

    unique_tt   = sorted({k[0] for k in all_curve_keys})
    optim_types = sorted({k[1] for k in all_curve_keys})
    unique_loss_types = sorted({k[2] for k in all_curve_keys})

    # Pretty-print configuration
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

    # ------------------------------------------------------------------
    # Combined figure: validation error + training loss for every
    # (transition_time_s, loss_type) combination.
    # Layout: rows = unique_tt
    #         cols = 2 * len(unique_loss_types)
    #               [val_LT0 | loss_LT0 | val_LT1 | loss_LT1 | ...]
    # ------------------------------------------------------------------
    color_map = plt.get_cmap("tab10")
    optim_to_color = {opt: color_map(i % 10) for i, opt in enumerate(optim_types)}
    # Line style per loss_type (useful when overlaying)
    lt_linestyle = {lt: ls for lt, ls in zip(unique_loss_types, ["-", "--", "-.", ":"])}

    n_rows = len(unique_tt)
    n_cols = 2 * len(unique_loss_types)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 2.8 * n_rows),
        sharex=False, squeeze=False,
    )

    def _plot_curve_series(ax, series, color, linestyle, label):
        """Plot mean ± std of a list of (time_axis, values) pairs."""
        min_len = min(len(v[1]) for v in series)
        all_vals, common_time = [], None
        for time_axis, vals in series:
            all_vals.append(vals[:min_len])
            if common_time is None:
                common_time = time_axis[:min_len]
        if not all_vals or common_time is None:
            return
        vals_stack = np.stack(all_vals, axis=0)
        avg = np.mean(vals_stack, axis=0)
        std = np.std(vals_stack, axis=0)
        ax.plot(common_time, avg, color=color, linestyle=linestyle,
                linewidth=1.0, alpha=0.95, label=label)
        num_m = min(10, len(common_time))
        idxs = np.linspace(0, len(common_time) - 1, num=num_m, dtype=int)
        ax.errorbar(common_time[idxs], avg[idxs], yerr=std[idxs],
                    fmt="none", ecolor=color, elinewidth=1.0, capsize=3, alpha=0.7)

    def _add_transitions(ax, tt):
        trans = tt_transitions.get(tt, None)
        if trans is None:
            return
        for t_start, t_end in trans:
            ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
            if t_start != t_end:
                ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

    for row_i, tt in enumerate(unique_tt):
        for lt_i, lt in enumerate(unique_loss_types):
            # Column order: loss first, then validation error
            ax_loss = axes[row_i, lt_i * 2]
            ax_val  = axes[row_i, lt_i * 2 + 1]

            # Column header titles on top row only (no tt — shown via row annotation)
            if row_i == 0:
                ax_loss.set_title(f"{lt} — Training loss")
                ax_val.set_title(f"{lt} — Validation error")

            ax_val.axhline(y=1.0, color="dimgray", linestyle="--", linewidth=1.0, alpha=0.6)
            _add_transitions(ax_val, tt)
            _add_transitions(ax_loss, tt)

            for optim in optim_types:
                color = optim_to_color[optim]
                label = optim.replace("_", " ")

                val_key  = (tt, optim, lt)
                loss_key = (tt, optim, lt)

                if val_key in curves and curves[val_key]:
                    _plot_curve_series(ax_val, curves[val_key], color, "-", label)

                if loss_key in loss_curves and loss_curves[loss_key]:
                    series = loss_curves[loss_key]
                    # clip for log scale
                    clipped = [(ta, np.clip(v, 1e-8, None)) for ta, v in series]
                    _plot_curve_series(ax_loss, clipped, color, "-", label)

            ax_loss.set_yscale("log")

            if row_i == n_rows - 1:
                ax_val.set_xlabel("Time [s]")
                ax_loss.set_xlabel("Time [s]")


        # Horizontal row label on the leftmost axis — separate from the y-axis label
        axes[row_i, 0].annotate(
            f"Transition time: {tt} s",
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-70, 0), textcoords="offset points",
            ha="right", va="center", rotation=0,
            fontsize=9,
            annotation_clip=False,
        )

        # Legend on the first (loss) subplot of the first loss type, first row only
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right", fontsize=7)

    plt.tight_layout()
    plt.subplots_adjust(left=0.12)

    # ------------------------------------------------------------------
    # Condensed checkpoint figure
    # Layout: rows = unique_loss_types
    #         cols = n_checkpoints  (taken from representative optimizer)
    # All optimizers' EQ responses are overlaid per cell.
    # Shared responses (desired, LEM, total) from the first available optimizer.
    # ------------------------------------------------------------------
    if checkpoint_examples:
        # checkpoint_examples: {loss_type: {optim_type: [cp, ...]}}
        # Normalize: handle old format ({optim_type: [cp]}) transparently
        if not isinstance(next(iter(checkpoint_examples.values())), dict):
            # Old single-level format — wrap it in a dict keyed by a placeholder loss_type
            checkpoint_examples = {"(default)": checkpoint_examples}

        cp_loss_types = sorted(checkpoint_examples.keys())
        # Find max number of checkpoints across all (lt, optim) combos
        n_cp_cols = max(
            len(cps)
            for lt_cps in checkpoint_examples.values()
            for cps in lt_cps.values()
        )
        n_cp_rows = len(cp_loss_types)

        fig2, axes2 = plt.subplots(
            n_cp_rows, n_cp_cols,
            figsize=(3.5 * n_cp_cols, 3.5 * n_cp_rows),
            sharex=True, sharey=True, squeeze=False,
        )

        eq_colors = {opt: color_map(i % 10) for i, opt in enumerate(optim_types)}

        color_desired = "black"
        color_lem     = "black"

        for row_i, lt in enumerate(cp_loss_types):
            by_optim = checkpoint_examples[lt]
            all_optims_sorted = sorted(by_optim.keys())

            # reference (shared background responses) from the first optimizer
            ref_optim = all_optims_sorted[0]
            ref_cps   = by_optim[ref_optim]

            for col_i in range(n_cp_cols):
                ax = axes2[row_i, col_i]

                # Shared reference from first optimizer
                if col_i < len(ref_cps):
                    ref_cp = ref_cps[col_i]
                    freqs_log  = np.asarray(ref_cp["freqs_log"],    dtype=float)
                    H_desired  = np.asarray(ref_cp["H_desired_db"], dtype=float)
                    H_lem      = np.asarray(ref_cp["H_lem_db"],     dtype=float)
                    H_total_ref = np.asarray(ref_cp["H_total_db"],  dtype=float)
                    sr_cp = float(ref_cp.get("sr", cfg.get("sample_rate", 48000)))
                    t_s   = float(ref_cp.get("time_s", 0.0))

                    ax.plot(freqs_log, H_desired, color=color_desired,
                            linewidth=1.4, linestyle="-", alpha=1.0, label="Desired", zorder=5)
                    ax.plot(freqs_log, H_lem, color=color_lem,
                            linewidth=1.0, linestyle="--", alpha=0.8, label="LEM (no EQ)", zorder=4)

                    if row_i == 0:
                        ax.set_title(f"t = {t_s:.1f} s")

                # Per-optimizer compensated response (EQ + LEM) overlays
                for opt in all_optims_sorted:
                    opt_cps = by_optim[opt]
                    if col_i >= len(opt_cps):
                        continue
                    cp = opt_cps[col_i]
                    freqs_log_o = np.asarray(cp["freqs_log"], dtype=float)
                    H_total_opt = cp.get("H_total_db", None)
                    if H_total_opt is not None:
                        ax.plot(freqs_log_o, np.asarray(H_total_opt, dtype=float),
                                color=eq_colors.get(opt, "gray"),
                                linewidth=1.0, alpha=0.55,
                                label=opt.replace("_", " "), zorder=3)

                ax.set_xscale("log")
                ax.set_ylim(-20, 20)
                ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
                if row_i == n_cp_rows - 1:
                    ax.set_xlabel("Frequency [Hz]")
                if col_i == 0:
                    ax.set_ylabel(f"{lt}\nMagnitude [dB]")

        # Single legend in top-left cell
        handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
        if handles2:
            axes2[0, 0].legend(handles2, labels2, loc="best", fontsize=7)

        plt.tight_layout()

    plt.show()


def main() -> None:
    # Select the experiment to plot here
    experiment_name = "experiment_04_run_03"

    # Project root (same convention as experiment_04.py)
    root = Path(__file__).resolve().parents[1]

    cfg, plot1_data = load_results(experiment_name, root)
    print(f"Experiment name: {experiment_name}")
    plot_results(cfg, plot1_data)


if __name__ == "__main__":
    main()
