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
        "axes.titlesize": 12,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.titlesize": 13,
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


def _latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace(" ", r"\ ")


def _format_input_label(input_signal) -> str:
    """Map an input-signal identifier to a compact, readable label."""
    s = str(input_signal)
    if s.startswith("white_noise"):
        return s
    p = Path(s)
    if p.suffix:
        return p.stem
    return s


def _select_series_for_averaging(
    series,
    n_remove_highest_mean_curves: int = 0,
    run_labels=None,
    report_context: str = None,
):
    """Optionally remove n runs with the highest time-mean value."""
    if not series:
        return series

    n_remove = max(0, int(n_remove_highest_mean_curves))
    if n_remove == 0:
        return series

    min_len = min(len(v[1]) for v in series)
    if min_len == 0 or len(series) <= 1:
        return series

    # Keep at least one run to avoid empty aggregation.
    n_remove = min(n_remove, len(series) - 1)
    if n_remove == 0:
        return series

    mean_per_run = []
    for _, vals in series:
        run_vals = np.asarray(vals[:min_len], dtype=float)
        mean_per_run.append(np.nanmean(run_vals))

    means = np.asarray(mean_per_run, dtype=float)
    idx_sorted_desc = np.argsort(-means)
    idx_remove_list = [int(i) for i in idx_sorted_desc[:n_remove]]
    idx_remove = set(idx_remove_list)

    if report_context:
        removed_labels = []
        for i in idx_remove_list:
            if run_labels is not None and i < len(run_labels):
                removed_labels.append(str(run_labels[i]))
            else:
                removed_labels.append(f"run_{i}")
        if removed_labels:
            print(f"[curve-filter] {report_context} -> removed: {', '.join(removed_labels)}")

    return [s for i, s in enumerate(series) if i not in idx_remove]


def _plot_mean_std(
    ax,
    series,
    color,
    linestyle,
    label,
    n_remove_highest_mean_curves: int = 0,
    run_labels=None,
    report_context: str = None,
):
    """Plot mean +- std from list of (time_axis, values)."""
    series = _select_series_for_averaging(
        series,
        n_remove_highest_mean_curves,
        run_labels=run_labels,
        report_context=report_context,
    )
    if not series:
        return
    min_len = min(len(v[1]) for v in series)
    all_vals, common_time = [], None
    for time_axis, vals in series:
        if not np.isnan(vals).any():
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


def _plot_dual_scenario_validation(root: Path, n_remove_highest_mean_curves: int = 0) -> None:
    scenario_to_experiment = {
        "Moving listener position (all songs)": "experiment_04_ALL_SONGS_MOVING_POSITION",
        "Moving person (all songs)": "experiment_04_ALL_SONGS_MOVING_PERSON",
        "Moving listener position (white noise)": "experiment_04_WHITE_NOISE_MOVING_POSITION",
        "Moving person (white noise)": "experiment_04_WHITE_NOISE_MOVING_PERSON",
    }

    loaded = {}
    for scenario_name, experiment_name in scenario_to_experiment.items():
        try:
            _, plot_data = load_results(experiment_name, root)
            loaded[scenario_name] = plot_data
        except FileNotFoundError as exc:
            print(f"Skipping dual-scenario plot: {exc}")
            return

    all_tt = sorted(
        {
            key[0]
            for plot_data in loaded.values()
            for key in plot_data.get("curves", {}).keys()
        }
    )
    if not all_tt:
        print("Skipping dual-scenario plot: no validation curves available.")
        return

    # Build unified optimizer/loss spaces for consistent colors/styles.
    all_optim = sorted(
        {
            key[1]
            for plot_data in loaded.values()
            for key in plot_data.get("curves", {}).keys()
        }
    )
    all_loss = sorted(
        {
            key[2]
            for plot_data in loaded.values()
            for key in plot_data.get("curves", {}).keys()
        }
    )

    color_map = plt.get_cmap("tab10")
    optim_to_color = {opt: color_map(i % 10) for i, opt in enumerate(all_optim)}
    lt_linestyle = {lt: ls for lt, ls in zip(all_loss, ["-", "--", "-.", ":"])}

    n_rows = len(all_tt)
    n_cols = len(scenario_to_experiment)

    # Dedicated left label column for transition-time annotations.
    fig = plt.figure(figsize=(2.5 * n_cols + 1.5, 1.5 * n_rows + 0.8))
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
            width_ratios=[0.22] + [1.0] * n_cols,
    )
    label_axes = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
    for ax_label in label_axes:
        ax_label.set_axis_off()
    axes = np.array([[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)]
                     for r in range(n_rows)])

    scenario_items = list(scenario_to_experiment.items())
    for row_i, tt in enumerate(all_tt):
        for col_i, (scenario_name, _) in enumerate(scenario_items):
            ax = axes[row_i, col_i]
            plot_data = loaded[scenario_name]
            curves = plot_data.get("curves", {})
            tt_transitions = plot_data.get("tt_transitions", {})
            scenario_input_signals = plot_data.get("input_signals", None)
            run_labels = [
                _format_input_label(sig) for sig in scenario_input_signals
            ] if scenario_input_signals else None

            if row_i == 0:
                ax.set_title(r"$\mathrm{" + _latex_escape(scenario_name) + r"}$")

            ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)

            trans = tt_transitions.get(tt, None)
            if trans is not None:
                for t_start, t_end in trans:
                    ax.axvline(x=t_start, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                    if t_start != t_end:
                        ax.axvline(x=t_end, color="0.2", linestyle="--", linewidth=1.0, alpha=0.9)
                        ax.axvspan(t_start, t_end, color="0.8", alpha=0.3)

            for lt in all_loss:
                for optim in all_optim:
                    key = (tt, optim, lt)
                    if key not in curves or not curves[key]:
                        continue

                    legend_label = r"$\mathrm{" + _latex_escape(optim.replace("_", " ")) + r"}$"
                    _plot_mean_std(
                        ax,
                        curves[key],
                        color=optim_to_color[optim],
                        linestyle=lt_linestyle.get(lt, "-"),
                        label=legend_label,
                        n_remove_highest_mean_curves=n_remove_highest_mean_curves,
                        run_labels=run_labels,
                        report_context=f"{scenario_name} | tt={tt}, optim={optim}, loss={lt}",
                    )

            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
            ax.set_ylim(-0.2, 1.5)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["0", "1"])
            is_left_col = (col_i == 0)
            is_bottom_row = (row_i == n_rows - 1)
            ax.tick_params(
                axis="x",
                which="both",
                bottom=is_bottom_row,
                labelbottom=is_bottom_row,
                top=False,
                labeltop=False,
            )
            ax.tick_params(
                axis="y",
                which="both",
                left=is_left_col,
                labelleft=is_left_col,
                right=False,
                labelright=False,
            )
            if row_i == n_rows - 1:
                ax.set_xlabel(r"$\mathrm{Time\ [s]}$")

        label_axes[row_i].text(
            0.5, 0.5,
            f"Transition\ntime: {tt} s",
            transform=label_axes[row_i].transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    handles, labels = axes[0, -1].get_legend_handles_labels()
    if handles:
        # De-duplicate labels while preserving order.
        unique_h, unique_l = [], []
        for h, l in zip(handles, labels):
            if l in unique_l:
                continue
            unique_h.append(h)
            unique_l.append(l)
        axes[0, -1].legend(
            unique_h,
            unique_l,
            loc="upper right",
            fontsize=7,
            frameon=True,
            borderpad=0.2,
            labelspacing=0.2,
            handlelength=1.2,
            handletextpad=0.3,
            columnspacing=0.6,
            borderaxespad=0.2,
        )

    #fig.suptitle(r"$\mathrm{Validation\ Error\ Comparison:\ Moving\ Position\ vs\ Moving\ Person}$")
    fig.tight_layout(pad=0.3, w_pad=0.1, h_pad=0.9)


def plot_results(cfg: dict, plot1_data: dict, n_remove_highest_mean_curves: int = 0) -> None:
    curves = plot1_data["curves"]
    loss_curves = plot1_data.get("loss_curves", {})
    compute_time_stats = plot1_data.get("compute_time_stats", {})
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
    run_labels = [_format_input_label(sig) for sig in input_signals] if input_signals else None

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
    # Compute-time table: rows = transition_time_s, cols = optimizer
    # Each cell reports average compute time per frame [s/frame]
    # ------------------------------------------------------------------
    if compute_time_stats:
        compute_time_table = np.full((len(unique_tt), len(optim_types)), np.nan, dtype=float)
        for tt_i, tt in enumerate(unique_tt):
            for opt_i, opt in enumerate(optim_types):
                stats = compute_time_stats.get((tt, opt), None)
                if not stats:
                    continue
                total_frames = int(stats.get("total_frames", 0))
                total_time_s = float(stats.get("total_time_s", 0.0))
                if total_frames > 0:
                    compute_time_table[tt_i, opt_i] = total_time_s / total_frames

        print("Average compute time per frame [s/frame] (rows=transition_time_s, cols=optimizer):")
        header = "transition_time_s" + "".join(f"\t{opt}" for opt in optim_types)
        print(header)
        for tt_i, tt in enumerate(unique_tt):
            row_vals = []
            for opt_i in range(len(optim_types)):
                v = compute_time_table[tt_i, opt_i]
                row_vals.append(f"{v:.6f}" if np.isfinite(v) else "nan")
            print(f"{tt}\t" + "\t".join(row_vals))
        print()

        # Plot timing table in a dedicated figure for quick visual comparison
        fig_time, ax_time = plt.subplots(
            figsize=(max(6.0, 1.2 * len(optim_types) + 2.0), max(2.5, 0.6 * len(unique_tt) + 1.8))
        )
        ax_time.axis("off")
        table_cell_text = [
            [f"{v:.6f}" if np.isfinite(v) else "-" for v in row]
            for row in compute_time_table
        ]
        table_row_labels = [f"tt={tt}s" for tt in unique_tt]
        timing_table = ax_time.table(
            cellText=table_cell_text,
            rowLabels=table_row_labels,
            colLabels=optim_types,
            cellLoc="center",
            loc="center",
        )
        timing_table.auto_set_font_size(False)
        timing_table.set_fontsize(8)
        timing_table.scale(1.0, 1.2)
        ax_time.set_title("Average compute time per frame [s/frame]")
        fig_time.tight_layout()

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
    # Extra label column (index 0) with half the width of a data column
    fig = plt.figure(figsize=(2.5 * n_cols + 1.2, 2 * n_rows))
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[0.5] + [1.0] * n_cols,
    )
    # Hidden axes in column 0 — used only for row labels
    label_axes = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
    for _ax in label_axes:
        _ax.set_axis_off()
    # Data axes occupy columns 1..n_cols
    axes = np.array([[fig.add_subplot(gs[r, c + 1]) for c in range(n_cols)]
                     for r in range(n_rows)])

    def _plot_curve_series(ax, series, color, linestyle, label, report_context: str = None):
        """Plot mean ± std of a list of (time_axis, values) pairs."""
        series = _select_series_for_averaging(
            series,
            n_remove_highest_mean_curves,
            run_labels=run_labels,
            report_context=report_context,
        )
        if not series:
            return
        min_len = min(len(v[1]) for v in series)
        all_vals, common_time = [], None
        for time_axis, vals in series:
            if not np.isnan(vals).any(): # Only append if no NaNs appear
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

            ax_val.axhline(y=1.0, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
            _add_transitions(ax_val, tt)
            _add_transitions(ax_loss, tt)

            for optim in optim_types:
                color = optim_to_color[optim]
                label = optim.replace("_", " ")

                val_key  = (tt, optim, lt)
                loss_key = (tt, optim, lt)

                if val_key in curves and curves[val_key]:
                    _plot_curve_series(
                        ax_val,
                        curves[val_key],
                        color,
                        "-",
                        label,
                        report_context=f"tt={tt}, optim={optim}, loss={lt}",
                    )

                if loss_key in loss_curves and loss_curves[loss_key]:
                    series = loss_curves[loss_key]
                    # clip for log scale
                    clipped = [(ta, np.clip(v, 1e-8, None)) for ta, v in series]
                    _plot_curve_series(ax_loss, clipped, color, "-", label, report_context=None)

                # Set y-limits based on loss type
                if lt == 'FD-MSE':
                    ax_val.set_ylim(-0.1, 1.5)
                    ax_loss.set_ylim(1, 1e4)
                elif lt == 'TD-MSE':
                    ax_val.set_ylim(-0.1, 3.5)
            ax_loss.set_yscale("log")

            if row_i == n_rows - 1:
                ax_val.set_xlabel("Time [s]")
                ax_loss.set_xlabel("Time [s]")


        # Row label in its own dedicated column — always inside the figure
        label_axes[row_i].text(
            0.5, 0.5,
            f"Transition\ntime: {tt} s",
            transform=label_axes[row_i].transAxes,
            ha="center",
            va="center",
            fontsize=9,
        )

    # Legend on the first (loss) subplot of the first loss type, first row only
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, loc="upper right", fontsize=7)

    plt.tight_layout()

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
            figsize=(1.8 * n_cp_cols, 2.4 * n_cp_rows), # adjust checkpoint figure size
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
                                label="_nolegend_", zorder=3)

                ax.set_xscale("log")
                ax.set_ylim(-20, 20)
                ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
                if col_i == 0:
                    ax.set_ylabel("Magnitude [dB]", fontsize=10)

        # Single legend in top-left cell
        handles2, labels2 = axes2[0, 0].get_legend_handles_labels()
        if handles2:
            axes2[0, 0].legend(handles2, labels2, loc="best", fontsize=7)

        # Use manual figure text so x-label position is independent and tightly controlled.
        fig2.tight_layout()
        bottom_axes_y = min(ax.get_position().y0 for ax in axes2[-1, :])
        x_label_y = max(0.005, bottom_axes_y - 0.12)
        fig2.text(0.5, x_label_y, "Frequency [Hz]", ha="center", va="top", fontsize=10)


def main() -> None:
    # Select the experiment to plot here
    experiment_name = "experiment_04_ALL_SONGS_MOVING_PERSON"
    n_remove_highest_mean_curves = 2  # Set 0 to keep all curves, or n to remove the n highest-mean runs

    # Project root (same convention as experiment_04.py)
    root = Path(__file__).resolve().parents[1]

    cfg, plot1_data = load_results(experiment_name, root)
    print(f"Experiment name: {experiment_name}")
    plot_results(cfg, plot1_data, n_remove_highest_mean_curves=n_remove_highest_mean_curves)

    if experiment_name in {
        "experiment_04_ALL_SONGS_MOVING_POSITION",
        "experiment_04_ALL_SONGS_MOVING_PERSON",
        "experiment_04_WHITE_NOISE_MOVING_POSITION",
        "experiment_04_WHITE_NOISE_MOVING_PERSON",
    }:
        _plot_dual_scenario_validation(root, n_remove_highest_mean_curves=n_remove_highest_mean_curves)
    
    plt.show()


if __name__ == "__main__":
    main()
