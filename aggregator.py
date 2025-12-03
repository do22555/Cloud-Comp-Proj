# aggregator.py
"""Load results from workers and produce a Higgs-peak plot."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Must match analysis_common.py
xmin = 80.0
xmax = 250.0
step = 2.5 # this matches notebook - not sure if this matters really
BINS = np.arange(xmin, xmax + step, step)
BIN_CENTRES = 0.5 * (BINS[:-1] + BINS[1:])

# ATLAS Open Data plotting colours/definitions (from notebook)
SAMPLES = {
    "Data": {"color": "black"},
    r"Background $Z,t\bar{t},t\bar{t}+V,VVV$": {"color": "#6b59d3"},
    r"Background $ZZ^{*}$": {"color": "#ff0000"},
    r"Signal ($m_H$ = 125 GeV)": {"color": "#00cdff"},
}

OUTPUT_DIR = Path("/data/output")


# ---------------------------------------------------------------------------
# LOAD NPZ PARTIAL RESULTS
# ---------------------------------------------------------------------------
def load_partial_results():
    """
    Returns:
        hists[sample] = {
            "values": array,       # bin counts (summed over all files)
            "variances": array,    # per-bin variances (summed over all files)
        }
        events[sample] = int       # total events processed for that sample
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
            # np.savez stored 'sample' as a 0-d array of dtype=object
            if isinstance(sample, np.ndarray):
                sample = sample.item()
            sample = str(sample)

            values = f["values"]
            variances = f["variances"]
            ev = int(f["events"])

        if sample not in hists:
            hists[sample] = {
                "values": np.zeros_like(values, dtype=float),
                "variances": np.zeros_like(variances, dtype=float),
            }
            events[sample] = 0

        # Sum histograms bin-by-bin
        hists[sample]["values"] += values
        hists[sample]["variances"] += variances
        events[sample] += ev

    print(f"[aggregator] Loaded {len(npz_files)} partial histograms.")
    return hists, events


# ---------------------------------------------------------------------------
# MAIN PLOTTING LOGIC – CLOSE TO THE NOTEBOOK
# ---------------------------------------------------------------------------
def main():
    hists, events = load_partial_results()

    if "Data" not in hists:
        print("[aggregator] ERROR: No 'Data' histogram found.")
        return

    # -------------------------------
    # Extract data (Example 1 style)
    # -------------------------------
    data_y = hists["Data"]["values"]
    # use stored variances → Poisson sigma = sqrt(N) for unweighted events
    data_var = hists["Data"]["variances"]
    data_err = np.sqrt(data_var)

    # -------------------------------
    # Build MC stacks (if present)
    # -------------------------------
    mc_stack_vals = []
    mc_stack_vars = []
    mc_colors = []
    mc_labels = []

    signal_y = None
    signal_color = None

    for label, meta in SAMPLES.items():
        if label == "Data":
            continue

        # Signal handled separately
        if label.startswith("Signal"):
            if label in hists:
                signal_y = hists[label]["values"]
                signal_color = meta["color"]
            continue

        # Backgrounds
        if label in hists:
            mc_stack_vals.append(hists[label]["values"])
            mc_stack_vars.append(hists[label]["variances"])
            mc_colors.append(meta["color"])
            mc_labels.append(label)

    # Stack MC values and variances (if any MC present)
    if mc_stack_vals:
        stacked_vals = np.zeros_like(data_y, dtype=float)
        stacked_var = np.zeros_like(data_y, dtype=float)
        for vals, var in zip(mc_stack_vals, mc_stack_vars):
            stacked_vals += vals
            stacked_var += var
        mc_err = np.sqrt(stacked_var)
    else:
        stacked_vals = np.zeros_like(data_y, dtype=float)
        mc_err = np.zeros_like(data_y, dtype=float)

    # ------------------------------------------------------------------
    # PLOT (matches notebook layout as far as possible)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    # DATA POINTS WITH ERROR BARS (like notebook Example 1)
    ax.errorbar(
        BIN_CENTRES,
        data_y,
        yerr=data_err,
        fmt="ko",
        label="Data",
    )

    # BACKGROUND STACK
    bottom = np.zeros_like(data_y, dtype=float)
    for y, colour, label in zip(mc_stack_vals, mc_colors, mc_labels):
        ax.bar(
            BIN_CENTRES,
            y,
            bottom=bottom,
            width=step,
            label=label,
            color=colour,
        )
        bottom += y

    # MC STATISTICAL UNCERTAINTY BAND
    if mc_stack_vals:  # only draw if we actually have MC
        ax.bar(
            BIN_CENTRES,
            2 * mc_err,
            bottom=stacked_vals - mc_err,
            width=step,
            alpha=0.5,
            color="none",
            hatch="////",
            label="MC stat. unc.",
        )

    # SIGNAL (optional, if present)
    if signal_y is not None:
        ax.step(
            BIN_CENTRES,
            signal_y + stacked_vals,  # signal on top of backgrounds
            where="mid",
            color=signal_color,
            linewidth=2,
            label=r"Signal ($m_H=125$ GeV)",
        )

    # Aesthetics (close to the ATLAS tutorial style)
    ax.set_xlabel(r"4-lepton invariant mass $m_{4\ell}$ [GeV]", fontsize=13)
    ax.set_ylabel(f"Events / {step:.1f} GeV", fontsize=13)
    ax.set_xlim(xmin, xmax)

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
        which="both",
        direction="in",
        top=True,
        right=True,
    )

    ax.legend(frameon=False)
    plt.tight_layout()

    out_path = OUTPUT_DIR / "final_plot.png"
    fig.savefig(out_path)
    print(f"[aggregator] Plot saved to {out_path}")
    print("[aggregator] Done.")


if __name__ == "__main__":
    main()
