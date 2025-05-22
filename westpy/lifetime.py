def radiative_lifetime(westpp_file, ispin, band1, band2, n=None, e_zpl=None):
    """Computes radiative lifetime.

    Args:
        westpp_file (string): The JSON output file of Westpp
        ispin (int): spin index
        band1 (int): band index (transition from band1 to band2 is computed)
        band2 (int): band index (transition from band1 to band2 is computed)
        n (float or function of e_zpl): refractive index
        e_zpl (float): zero-phonon line (ZPL) energy (Rydberg)

    :Example:

    >>> from westpy import *
    >>> tau = radiative_lifetime("westpp.json",2,101,102,2.0,1.25)
    """
    #
    import json
    import numpy as np
    import scipy.constants as sc
    from westpy.units import Angstrom, Joule

    #
    assert ispin == 1 or ispin == 2
    #
    # read westpp
    with open(westpp_file, "r") as f:
        westpp_json = json.load(f)
    #
    gamma_only = westpp_json["system"]["basis"]["gamma_only"]
    nkstot = westpp_json["system"]["electron"]["nkstot"]
    lsda = westpp_json["system"]["electron"]["lsda"]
    nkpt = int(nkstot / 2) if lsda else nkstot
    ikpt0 = 1 + nkpt * (ispin - 1)
    ikpt1 = ikpt0 + nkpt
    #
    westpp_range = westpp_json["input"]["westpp_control"]["westpp_range"]
    nband = westpp_range[1] - westpp_range[0] + 1
    itrans = (band2 - westpp_range[0]) * nband + (band1 - westpp_range[0])
    #
    rr = np.zeros(3)
    for ikpt in range(ikpt0, ikpt1):
        label_k = "K" + "{:06d}".format(ikpt)
        #
        eig1 = westpp_json["output"]["D"][label_k]["energies"][band1 - 1]
        eig2 = westpp_json["output"]["D"][label_k]["energies"][band2 - 1]
        e_diff = eig2 - eig1
        #
        assert e_diff > 1.0e-8
        #
        wk = westpp_json["output"]["D"][label_k]["weight"]
        if not lsda:
            wk /= 2.0
        #
        re = np.zeros(3)
        im = np.zeros(3)
        if gamma_only:
            re[0] = westpp_json["output"]["D"][label_k]["dipole"]["x"][itrans]
            re[1] = westpp_json["output"]["D"][label_k]["dipole"]["y"][itrans]
            re[2] = westpp_json["output"]["D"][label_k]["dipole"]["z"][itrans]
        else:
            re[0] = westpp_json["output"]["D"][label_k]["dipole"]["x"]["re"][itrans]
            re[1] = westpp_json["output"]["D"][label_k]["dipole"]["y"]["re"][itrans]
            re[2] = westpp_json["output"]["D"][label_k]["dipole"]["z"]["re"][itrans]
            im[0] = westpp_json["output"]["D"][label_k]["dipole"]["x"]["im"][itrans]
            im[1] = westpp_json["output"]["D"][label_k]["dipole"]["y"]["im"][itrans]
            im[2] = westpp_json["output"]["D"][label_k]["dipole"]["z"]["im"][itrans]
        #
        for i in range(3):
            rr[i] += np.sqrt(re[i] ** 2 + im[i] ** 2) * wk / e_diff
    #
    rr_sq = sum(rr**2)
    #
    if e_zpl is None:
        e_zpl = e_diff
    assert e_zpl > 0.0
    #
    # refractive index
    if callable(n):
        refrac = n(e_zpl)
    elif isinstance(n, float):
        assert n >= 1.0
        refrac = n
    else:
        refrac = 1.0
    #
    # Bohr to m
    Meter = Angstrom * 1.0e10
    rr_sq /= Meter**2
    # Ry to J
    e_zpl /= Joule
    #
    # compute radiative lifetime using SI units
    tau = (3 * sc.epsilon_0 * sc.pi * (sc.c**3) * (sc.hbar**4)) / (
        refrac * (e_zpl**3) * (sc.e**2) * rr_sq
    )
    #
    return tau
