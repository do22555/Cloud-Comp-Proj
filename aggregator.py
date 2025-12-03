# aggregator.py

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Must match analysis_common.py
xmin = 80.0
xmax = 200.0
step = 2.0
BINS = np.arange(xmin, xmax + step, step)
BIN_CENTRES = 0.5 * (BINS[:-1] + BINS[1:])

# ATLAS Open Data plotting colours/definitions (from notebook)
SAMPLES = {
    "Data": {"color": "black"},
    r"Background $Z,t\bar{t},t\bar{t}+V,VVV$": {
        "color": "#6b59d3",
    },
    r"Background $ZZ^{*}$": {
        "color": "#ff0000",
    },
    r"Signal ($m_H$ = 125 GeV)": {
        "color": "#00cdff",
    },
}

OUTPUT_DIR = Path("/data/output")


# ---------------------------------------------------------------------------
# LOAD NPZ PARTIAL RESULTS
# ---------------------------------------------------------------------------
def load_partial_results():
    """
    Returns:
        hists[sample] = {"values": array, "variances": array}
        events[sample] = int
    """

    hists = {}
    events = {}

    for path in sorted(OUTPUT_DIR.glob("*.npz")):
        f = np.load(path, allow_pickle=True)

        sample = str(f["sample"])
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

    print(f"[aggregator] Loaded {len(list(OUTPUT_DIR.glob('*.npz')))} partial histograms.")
    return hists, events


# ---------------------------------------------------------------------------
# MAIN PLOTTING LOGIC – SAME AS THE NOTEBOOK
# ---------------------------------------------------------------------------
def main():
    hists, events = load_partial_results()

    if "Data" not in hists:
        print("[aggregator] ERROR: No data histogram found.")
        return

    # -------------------------------
    # Extract data
    # -------------------------------
    data_y = hists["Data"]["values"]
    data_err = np.sqrt(data_y)

    # -------------------------------
    # Build MC stacks
    # -------------------------------
    mc_stack = []
    mc_colors = []
    mc_labels = []

    signal_y = None
    signal_color = None

    for label, meta in SAMPLES.items():

        if label == "Data":
            continue

        if label.startswith("Signal"):
            if label in hists:
                signal_y = hists[label]["values"]
                signal_color = meta["color"]
            continue

        # backgrounds
        if label in hists:
            mc_stack.append(hists[label]["values"])
            mc_colors.append(meta["color"])
            mc_labels.append(label)

    # Stack MC
    stacked = np.zeros_like(data_y)
    for bg in mc_stack:
        stacked += bg

    mc_err = np.sqrt(stacked)

    # ------------------------------------------------------------------
    # PLOT (matches notebook layout, axes, style)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 8))

    # DATA POINTS
    ax.errorbar(
        BIN_CENTRES,
        data_y,
        yerr=data_err,
        fmt="ko",
        label="Data",
    )

    # BACKGROUND STACK
    bottom = np.zeros_like(data_y)
    for y, colour, label in zip(mc_stack, mc_colors, mc_labels):
        ax.bar(
            BIN_CENTRES,
            y,
            bottom=bottom,
            width=step,
            label=label,
            color=colour,
        )
        bottom += y

    # MC STAT UNCERTAINTY BAND
    ax.fill_between(
        BIN_CENTRES,
        stacked - mc_err,
        stacked + mc_err,
        step="mid",
        color="grey",
        alpha=0.4,
        label="MC stat. unc.",
    )

    # SIGNAL (line plot on top)
    if signal_y is not None:
        ax.step(
            BIN_CENTRES,
            signal_y,
            where="mid",
            color=signal_color,
            linewidth=2,
            label=r"Signal ($m_H=125$ GeV)",
        )

    # Aesthetics (ATLAS-like)
    ax.set_xlabel(r"$m_{4\ell}$ [GeV]", fontsize=16)
    ax.set_ylabel("Events", fontsize=16)
    ax.set_xlim(xmin, xmax)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()
    fig.savefig("/data/output/final_plot.png")

    print("[aggregator] Plot saved to /data/output/final_plot.png")
    print("[aggregator] Done.")


if __name__ == "__main__":
    main()
