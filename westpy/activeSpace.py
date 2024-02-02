def active_space(westpp_file, local_factor_thr=0.0, max_n_bands=10, max_i_band=0):
    """Defines active space based on localization factors of Kohn-Sham orbitals.

    :param westpp_file: The JSON output file of Westpp
    :type westpp_file: string
    :param local_factor_thr: Localization factor threshold
    :type local_factor_thr: float
    :param max_n_bands: Maximum number of bands in active space
    :type max_n_bands: int
    :param max_i_band: Maximum band index to be considered, for instance conduction band minimum
    :type max_i_band: int

    :Example:

    >>> from westpy import *
    >>> qp_bands = active_space("westpp.json",0.01,20)
    """
    #
    import json
    import numpy as np

    #
    with open(westpp_file, "r") as f:
        j = json.load(f)
    #
    lsda = j["system"]["electron"]["lsda"]
    westpp_range = j["input"]["westpp_control"]["westpp_range"]
    if max_i_band < 1:
        max_i_band = westpp_range[1]
    n_bands = max_i_band - westpp_range[0] + 1
    if max_n_bands > n_bands:
        max_n_bands = n_bands
    #
    if lsda:
        lf = np.zeros((n_bands), dtype=np.float64)
        ib = np.zeros((n_bands), dtype=int)
        jb = np.zeros((n_bands), dtype=int)
        #
        lf1 = np.array(j["output"]["L"]["K00001"]["local_factor"], dtype=np.float64)
        lf2 = np.array(j["output"]["L"]["K00002"]["local_factor"], dtype=np.float64)
        overlap_ab = j["output"]["L"]["overlap_ab"]
        #
        for ip, pair in enumerate(overlap_ab):
            if pair["ib"] <= max_i_band and pair["jb"] <= max_i_band:
                ib[ip] = pair["ib"]
                jb[ip] = pair["jb"]
                lf[ip] = max(lf1[ib[ip] - 1], lf2[jb[ip] - 1])
        #
        max_lf_ids = np.argsort(lf)[::-1][:max_n_bands]
        #
        zip_sort = [
            (ib[ip] + westpp_range[0] - 1, jb[ip] + westpp_range[0] - 1)
            for ip in max_lf_ids
            if lf[ip] > local_factor_thr
        ]
        zip_sort = sorted(zip_sort)
        qp_bands = list(map(list, zip(*zip_sort)))
    else:
        lf = np.array(
            j["output"]["L"]["K00001"]["local_factor"][:n_bands], dtype=np.float64
        )
        max_lf_ids = np.argsort(lf)[::-1][:max_n_bands]
        #
        qp_bands = [
            ib + westpp_range[0]
            for ib in max_lf_ids
            if ib + westpp_range[0] <= max_i_band and lf[ib] > local_factor_thr
        ]
        qp_bands = sorted(qp_bands)

    return qp_bands
