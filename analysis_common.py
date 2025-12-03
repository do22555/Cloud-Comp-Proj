# analysis_common.py
import os ###
import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()

# ATLAS Open Data
#from atlasopenmagic import install_from_environment
import atlasopenmagic as atom
os.environ["ATLAS_OPENMAGIC_NO_INSTALL"] = "1" ### This was a nightmare to figure out:
#install_from_environment() ### Skip atlas open magic install entirely + it improves runtime
atom.set_release("2025e-13tev-beta")

# Histogram binning
xmin = 80.0
xmax = 200.0
step = 2.0
BINS = np.arange(xmin, xmax + step, step)


# --------------------------------------------------------------------
# Build dataset list exactly like notebook
# --------------------------------------------------------------------
def build_samples():
    DEFS = {
        "Data": {"dids": ["data"]},
    }
    return atom.build_dataset(DEFS, skim="exactly4lep", protocol="https", cache=True)


# --------------------------------------------------------------------
#    PURE NUMPY HISTOGRAM MAKER
# --------------------------------------------------------------------
def make_hist(values):
    """
    values: 1D numpy array of 4-lepton masses
    returns dictionary safe for JSON/NPZ saving
    """
    hist_values, edges = np.histogram(values, bins=BINS)
    variances = hist_values.astype(float)  # Poisson variance = N

    return {
        "edges": edges,
        "values": hist_values,
        "variances": variances,
    }


# -----------------------------------------------------------------------------------
#    FILE PROCESSOR — modded dependencies to work around atlasopenmagic install
# -----------------------------------------------------------------------------------
# these are sent to aggregator.py
def process_file(file_url, sample_name="Data"):
    print(f"[worker] Opening {file_url}")

    # ROOT file → TTree "analysis"
    tree = uproot.open(file_url + ":analysis")

    # Needed branches (from notebook)
    branches = tree.arrays(
        ["lep_pt", "lep_eta", "lep_phi", "lep_e", "lep_type"],
        how=dict
    )

    # Build awkward Momentum4D objects (not entirely sure if this is required, worried to delete)
    leptons = ak.zip(
        {
            "pt": branches["lep_pt"],
            "eta": branches["lep_eta"],
            "phi": branches["lep_phi"],
            "E": branches["lep_e"],
        },
        with_name="Momentum4D"
    )

    # Need >= 4 leptons
    mask4 = ak.num(leptons) >= 4
    leptons = leptons[mask4]

    if len(leptons) == 0:
        return {
            "hist": make_hist(np.array([])),
            "events": 0,
            "file": file_url,
            "sample": sample_name,
        }

    # Compute 4-lepton invariant mass exactly like the notebook:
    p4 = leptons
    higgs_mass = (p4[:, 0] + p4[:, 1] + p4[:, 2] + p4[:, 3]).M
    higgs_mass = ak.to_numpy(higgs_mass)

    # Build simple numpy histogram
    hist = make_hist(higgs_mass)

    return {
        "hist": hist,
        "events": len(higgs_mass),
        "file": file_url,
        "sample": sample_name,
    }
