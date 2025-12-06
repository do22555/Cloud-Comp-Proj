# aggregator.py
"""Load results from workers and produce HZZ plots (3 figures)."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from analysis_common import (
    BINS,
    BIN_CENTRES,
    step,
    xmin,
    xmax,
    LUMI,
    SAMPLE_DEFS,
)

OUTPUT_DIR = Path("/data/output")

DATA_LABEL = "Data"
TTBAR_LABEL = r"Background $Z,t\bar{t},t\bar{t}+V,VVV$"
SIGNAL_LABEL = r"Signal ($m_H$ = 125 GeV)"


# ---------------------------------------------------------------------------
# LOAD NPZ PARTIAL RESULTS
# ---------------------------------------------------------------------------
def load_partial_results():
    """
    Returns:
        hists[sample] = {
            "values": array,       # bin sums (sum w)
            "variances": array,    # per-bin sum w^2
        }
        events[sample] = float     # total events (sum weights or counts)
    """

    hists = {}
    events = {}

    npz_files = sorted(OUTPUT_DIR.glob("*.npz"))
    if not npz_files:
        print("[aggregator] No partial .npz files found in /data/output.")
        return hists, events

    for path in npz_files:
        with np.load(path, allow_pickle=True) as f:
            sample = f["sample"]
            if isinstance(sample, np.ndarray):
                sample = sample.item()
            sample = str(sample)

            values = f["values"].astype(float)
            variances = f["variances"].astype(float)
            ev = float(f["events"])

        if sample not in hists:
            hists[sample] = {
                "values": np.zeros_like(values, dtype=float),
                "variances": np.zeros_like(variances, dtype=float),
            }
            events[sample] = 0.0

        hists[sample]["values"] += values
        hists[sample]["variances"] += variances
        events[sample] += ev

    print(f"[aggregator] Loaded {len(npz_files)} partial histograms.")
    return hists, events


# ---------------------------------------------------------------------------
# PLOT 1 – Example 1–style Data-only plot
# ---------------------------------------------------------------------------
def plot_example1_data(hists):
    if DATA_LABEL not in hists:
        print("[aggregator] Example1: no Data histogram found.")
        return

    data_vals = hists[DATA_LABEL]["values"]
    data_err = np.sqrt(hists[DATA_LABEL]["variances"])

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.errorbar(
        BIN_CENTRES,
        data_vals,
        yerr=data_err,
        fmt="ko",
        label="Data",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"4-lepton invariant mass $m_{4\ell}$ [GeV]", fontsize=13)
    ax.set_ylabel(f"Events / {step:.1f} GeV", fontsize=13)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.set_ylim(bottom=0, top=np.max(data_vals) * 1.6)
    ax.legend(frameon=False)

    plt.tight_layout()
    out_path = OUTPUT_DIR / "hzz_example1_data.png"
    fig.savefig(out_path)
    print(f"[aggregator] Example1 plot saved to {out_path}")


# ---------------------------------------------------------------------------
# PLOT 2 – Example 2–style ttbar/Z+... background with stat band
# ---------------------------------------------------------------------------
def plot_example2_ttbar(hists):
    if TTBAR_LABEL not in hists:
        print(f"[aggregator] Example2: no '{TTBAR_LABEL}' histogram found.")
        return

    vals = hists[TTBAR_LABEL]["values"]
    err = np.sqrt(hists[TTBAR_LABEL]["variances"])
    colour = SAMPLE_DEFS[TTBAR_LABEL]["color"]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Background as a histogram (bar)
    ax.bar(
        BIN_CENTRES,
        vals,
        width=step,
        color=colour,
        label=TTBAR_LABEL,
    )

    # Stat uncertainty band: sqrt(sum w^2)
    ax.bar(
        BIN_CENTRES,
        2 * err,
        bottom=vals - err,
        width=step,
        alpha=0.5,
        color="none",
        hatch="////",
        label="Stat. Unc.",
    )

    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"4-lepton invariant mass $m_{4\ell}$ [GeV]", fontsize=13)
    ax.set_ylabel(f"Events / {step:.1f} GeV", fontsize=13)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

    ax.legend(frameon=False)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "hzz_example2_ttbar.png"
    fig.savefig(out_path)
    print(f"[aggregator] Example2 plot saved to {out_path}")


# ---------------------------------------------------------------------------
# PLOT 3 – Final stacked Data vs MC + Signal plot
# ---------------------------------------------------------------------------
def plot_final_stack(hists):
    if DATA_LABEL not in hists:
        print("[aggregator] Final: no Data histogram found.")
        return

    # Data
    data_vals = hists[DATA_LABEL]["values"]
    data_err = np.sqrt(hists[DATA_LABEL]["variances"])

    # Backgrounds & signal
    mc_x = []
    mc_colors = []
    mc_labels = []
    mc_vars = []

    signal_vals = None
    signal_color = None

    for name, meta in SAMPLE_DEFS.items():
        if name == DATA_LABEL:
            continue

        if name == SIGNAL_LABEL:
            if name in hists:
                signal_vals = hists[name]["values"]
                signal_color = meta["color"]
            continue

        # backgrounds
        if name in hists:
            mc_x.append(hists[name]["values"])
            mc_vars.append(hists[name]["variances"])
            mc_colors.append(meta["color"])
            mc_labels.append(name)

    # Build stacked background and total MC variance
    if mc_x:
        stacked_vals = np.zeros_like(mc_x[0])
        stacked_var = np.zeros_like(mc_vars[0])
        for vals, var in zip(mc_x, mc_vars):
            stacked_vals += vals
            stacked_var += var
        mc_err = np.sqrt(stacked_var)
    else:
        stacked_vals = np.zeros_like(data_vals)
        mc_err = np.zeros_like(data_vals)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Data points
    ax.errorbar(
        BIN_CENTRES,
        data_vals,
        yerr=data_err,
        fmt="ko",
        label="Data",
    )

    # Background stack
    bottom = np.zeros_like(data_vals, dtype=float)
    for vals, colour, label in zip(mc_x, mc_colors, mc_labels):
        ax.bar(
            BIN_CENTRES,
            vals,
            width=step,
            bottom=bottom,
            color=colour,
            label=label,
        )
        bottom += vals

    # Signal on top of backgrounds
    if signal_vals is not None:
        ax.step(
            BIN_CENTRES,
            stacked_vals + signal_vals,
            where="mid",
            color=signal_color,
            linewidth=2,
            label=SIGNAL_LABEL,
        )

    # MC stat. uncertainty band
    if mc_x:
        ax.bar(
            BIN_CENTRES,
            2 * mc_err,
            bottom=stacked_vals - mc_err,
            width=step,
            alpha=0.5,
            color="none",
            hatch="////",
            label="Stat. Unc.",
        )

    # Axes / labels (as in notebook final)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(r"4-lepton invariant mass $m_{4\ell}$ [GeV]", fontsize=13)
    ax.set_ylabel(f"Events / {step:.1f} GeV", fontsize=13)

    ax.set_ylim(bottom=0, top=np.max(data_vals) * 2.0)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which="both", direction="in", top=True, right=True)

    # ATLAS text
    plt.text(
        0.1,
        0.93,
        "ATLAS Open Data",
        transform=ax.transAxes,
        fontsize=16,
    )
    plt.text(
        0.1,
        0.88,
        "for education",
        transform=ax.transAxes,
        fontsize=12,
        style="italic",
    )
    lumi_used = f"{LUMI:.1f}"
    plt.text(
        0.1,
        0.82,
        rf"$\sqrt{{s}}$=13 TeV,$\int$L dt = {lumi_used} fb$^{{-1}}$",
        transform=ax.transAxes,
        fontsize=16,
    )
    plt.text(
        0.1,
        0.76,
        r"$H \rightarrow ZZ^* \rightarrow 4\ell$",
        transform=ax.transAxes,
        fontsize=16,
    )

    ax.legend(frameon=False, fontsize=14)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "hzz_final_stack.png"
    fig.savefig(out_path)
    print(f"[aggregator] Final stacked plot saved to {out_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    hists, events = load_partial_results()
    if not hists:
        print("[aggregator] Nothing to plot.")
        return

    # Produce all three figures
    plot_example1_data(hists)
    plot_example2_ttbar(hists)
    plot_final_stack(hists)

    print("[aggregator] Done.")


if __name__ == "__main__":
    main()
