# analysis_common.py

import os
import numpy as np
import awkward as ak
import uproot
import vector

vector.register_awkward()

# ----------------------------------------------------------------------
# ATLAS Open Data / atlasopenmagic
# ----------------------------------------------------------------------
# IMPORTANT: disable atlasopenmagic's environment auto-install
os.environ["ATLAS_OPENMAGIC_NO_INSTALL"] = "1"

import atlasopenmagic as atom  # noqa: E402

# Energy units (match notebook)
MeV = 0.001
GeV = 1.0

# Full-release luminosity
LUMI = 36.6  # fb^-1
FRACTION = 1.0  # can be <1.0 to speed up

atom.set_release("2025e-13tev-beta")

# ----------------------------------------------------------------------
# Sample definitions (as in the notebook)
# ----------------------------------------------------------------------
SAMPLE_DEFS = {
    r"Data": {"dids": ["data"]},
    r"Background $Z,t\bar{t},t\bar{t}+V,VVV$": {
        "dids": [
            410470,
            410155,
            410218,
            410219,
            412043,
            364243,
            364242,
            364246,
            364248,
            700320,
            700321,
            700322,
            700323,
            700324,
            700325,
        ],
        "color": "#6b59d3",  # purple
    },
    r"Background $ZZ^{*}$": {
        "dids": [700600],
        "color": "#ff0000",  # red
    },
    r"Signal ($m_H$ = 125 GeV)": {
        "dids": [
            345060,
            346228,
            346310,
            346311,
            346312,
            346340,
            346341,
            346342,
        ],
        "color": "#00cdff",  # light blue
    },
}


def build_samples():
    """Build atlasopenmagic dataset dictionary, like the notebook."""
    return atom.build_dataset(
        SAMPLE_DEFS,
        skim="exactly4lep",
        protocol="https",
        cache=True,
    )


# ----------------------------------------------------------------------
# Binning (match FINAL ANALYSIS binning)
# ----------------------------------------------------------------------
xmin = 80.0 * GeV
xmax = 250.0 * GeV
step = 2.5 * GeV

BINS = np.arange(xmin, xmax + step, step)
BIN_CENTRES = 0.5 * (BINS[:-1] + BINS[1:])

# ----------------------------------------------------------------------
# Physics variable lists (match notebook)
# ----------------------------------------------------------------------
VARIABLES = [
    "lep_pt",
    "lep_eta",
    "lep_phi",
    "lep_e",
    "lep_charge",
    "lep_type",
    "trigE",
    "trigM",
    "lep_isTrigMatched",
    "lep_isLooseID",
    "lep_isMediumID",
    "lep_isLooseIso",
    "lep_type",
]

WEIGHT_VARS = [
    "filteff",
    "kfac",
    "xsec",
    "mcWeight",
    "ScaleFactor_PILEUP",
    "ScaleFactor_ELE",
    "ScaleFactor_MUON",
    "ScaleFactor_LepTRIGGER",
]


# ----------------------------------------------------------------------
# Cuts & mass/weight calculations (copied from notebook logic)
# ----------------------------------------------------------------------
def cut_lep_type(lep_type):
    sum_lep_type = (
        lep_type[:, 0]
        + lep_type[:, 1]
        + lep_type[:, 2]
        + lep_type[:, 3]
    )
    lep_type_cut_bool = (sum_lep_type != 44) & (sum_lep_type != 48) & (sum_lep_type != 52)
    return lep_type_cut_bool  # True means we REMOVE these events


def cut_lep_charge(lep_charge):
    sum_lep_charge = (
        lep_charge[:, 0]
        + lep_charge[:, 1]
        + lep_charge[:, 2]
        + lep_charge[:, 3]
        != 0
    )
    return sum_lep_charge  # True means we REMOVE these events


def calc_mass(lep_pt, lep_eta, lep_phi, lep_e):
    p4 = vector.zip({"pt": lep_pt, "eta": lep_eta, "phi": lep_phi, "E": lep_e})
    invariant_mass = (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M
    return invariant_mass


def cut_trig_match(lep_trigmatch):
    trigmatch = lep_trigmatch
    cut1 = ak.sum(trigmatch, axis=1) >= 1
    return cut1


def cut_trig(trigE, trigM):
    return trigE | trigM


def ID_iso_cut(IDel, IDmu, isoel, isomu, pid):
    thispid = pid
    return (
        ak.sum(
            ((thispid == 13) & IDmu & isomu) | ((thispid == 11) & IDel & isoel),
            axis=1,
        )
        == 4
    )


def calc_weight(weight_variables, events, lumi=LUMI):
    """
    Monte Carlo total weight, identical logic to notebook Example 2 / Final.
    """
    total_weight = lumi * 1000.0 / events["sum_of_weights"]
    for variable in weight_variables:
        total_weight = total_weight * abs(events[variable])
    return total_weight


# ----------------------------------------------------------------------
# Histogram helper (works for weighted & unweighted)
# ----------------------------------------------------------------------
def make_hist(values, weights=None):
    """
    Create numpy histogram and sumw2 variance.

    values: 1D numpy array of masses
    weights: 1D numpy array of weights (or None for unweighted)
    """
    if values.size == 0:
        hist_values = np.zeros(len(BINS) - 1, dtype=float)
        variances = np.zeros_like(hist_values)
        return {
            "edges": BINS,
            "values": hist_values,
            "variances": variances,
        }

    if weights is None:
        weights = np.ones_like(values, dtype=float)

    hist_values, edges = np.histogram(values, bins=BINS, weights=weights)
    sumw2, _ = np.histogram(values, bins=BINS, weights=weights**2)

    return {
        "edges": edges,
        "values": hist_values.astype(float),
        "variances": sumw2.astype(float),
    }


# ----------------------------------------------------------------------
# FILE PROCESSOR: full final-analysis logic per file
# ----------------------------------------------------------------------
def _process_data_tree(tree):
    """Apply final-analysis cuts to a Data file, return masses & weights."""
    masses = []
    weights = []

    for data in tree.iterate(
        VARIABLES + ["lep_n"],
        library="ak",
        entry_stop=tree.num_entries * FRACTION,
    ):
        # Trigger cuts
        data = data[cut_trig(data.trigE, data.trigM)]
        data = data[cut_trig_match(data.lep_isTrigMatched)]

        # Leading lepton pT
        data["leading_lep_pt"] = data["lep_pt"][:, 0]
        data["sub_leading_lep_pt"] = data["lep_pt"][:, 1]
        data["third_leading_lep_pt"] = data["lep_pt"][:, 2]
        data["last_lep_pt"] = data["lep_pt"][:, 3]

        data = data[data["leading_lep_pt"] > 20]
        data = data[data["sub_leading_lep_pt"] > 15]
        data = data[data["third_leading_lep_pt"] > 10]

        # ID + isolation
        data = data[
            ID_iso_cut(
                data.lep_isLooseID,
                data.lep_isMediumID,
                data.lep_isLooseIso,
                data.lep_isLooseIso,
                data.lep_type,
            )
        ]

        # Lepton type & charge cuts
        lep_type = data["lep_type"]
        data = data[~cut_lep_type(lep_type)]
        lep_charge = data["lep_charge"]
        data = data[~cut_lep_charge(lep_charge)]

        # Invariant mass
        data["mass"] = calc_mass(
            data["lep_pt"],
            data["lep_eta"],
            data["lep_phi"],
            data["lep_e"],
        )

        m = ak.to_numpy(data["mass"])
        if m.size == 0:
            continue

        masses.append(m)
        weights.append(np.ones_like(m, dtype=float))

    if not masses:
        return np.array([]), np.array([])

    return np.concatenate(masses), np.concatenate(weights)


def _process_mc_tree(tree):
    """Apply final-analysis cuts to an MC file, return masses & MC weights."""
    masses = []
    weights = []

    for data in tree.iterate(
        VARIABLES + WEIGHT_VARS + ["sum_of_weights", "lep_n"],
        library="ak",
        entry_stop=tree.num_entries * FRACTION,
    ):
        # Trigger cuts
        data = data[cut_trig(data.trigE, data.trigM)]
        data = data[cut_trig_match(data.lep_isTrigMatched)]

        # Leading lepton pT
        data["leading_lep_pt"] = data["lep_pt"][:, 0]
        data["sub_leading_lep_pt"] = data["lep_pt"][:, 1]
        data["third_leading_lep_pt"] = data["lep_pt"][:, 2]
        data["last_lep_pt"] = data["lep_pt"][:, 3]

        data = data[data["leading_lep_pt"] > 20]
        data = data[data["sub_leading_lep_pt"] > 15]
        data = data[data["third_leading_lep_pt"] > 10]

        # ID + isolation
        data = data[
            ID_iso_cut(
                data.lep_isLooseID,
                data.lep_isMediumID,
                data.lep_isLooseIso,
                data.lep_isLooseIso,
                data.lep_type,
            )
        ]

        # Lepton type & charge cuts
        lep_type = data["lep_type"]
        data = data[~cut_lep_type(lep_type)]
        lep_charge = data["lep_charge"]
        data = data[~cut_lep_charge(lep_charge)]

        # Invariant mass
        data["mass"] = calc_mass(
            data["lep_pt"],
            data["lep_eta"],
            data["lep_phi"],
            data["lep_e"],
        )

        # MC weights
        data["totalWeight"] = calc_weight(WEIGHT_VARS, data, lumi=LUMI)

        m = ak.to_numpy(data["mass"])
        w = ak.to_numpy(data["totalWeight"])

        if m.size == 0:
            continue

        masses.append(m)
        weights.append(w)

    if not masses:
        return np.array([]), np.array([])

    return np.concatenate(masses), np.concatenate(weights)


def process_file(file_url, sample_name="Data"):
    """
    Worker entry point: open ROOT file, apply notebook-style final-analysis logic,
    and return a dict with a pure-numpy histogram.
    """
    print(f"[worker] Opening {file_url} for sample '{sample_name}'")

    tree = uproot.open(file_url + ":analysis")

    # Decide data vs MC from sample_name
    is_data = (sample_name == "Data")

    if is_data:
        masses, weights = _process_data_tree(tree)
    else:
        masses, weights = _process_mc_tree(tree)

    if masses.size == 0:
        hist = make_hist(np.array([]), None)
        events = 0.0
    else:
        hist = make_hist(masses, weights)
        # For logging: use sum of weights for MC, count for data
        events = float(weights.sum()) if not is_data else float(len(masses))

    return {
        "hist": hist,
        "events": events,
        "file": file_url,
        "sample": sample_name,
    }
