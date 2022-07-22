from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np
from scipy.special import erf, erfc
from lxml import etree
from .misc import find_index, find_indices, parse_one_value, parse_many_values

_fxc_template = """\
<?xml version="1.0"?>
<?iotk version="1.2.0"?>
<?iotk file_version="1.0"?>
<?iotk binary="F"?>
<?iotk qe_syntax="F"?>
<Root>
  <FXC_PDEPBASIS>
    <ndim type="integer" size="1">
       {npdep}
    </ndim>
    <fxc type="real" size="{fxcdim}">
{fxcdata}
    </fxc>
  </FXC_PDEPBASIS>
</Root>

"""


class WstatOutput:
    def __init__(self, savepath="."):
        self.savepath = savepath
        self.js = json.load(open("{}/wstat.json".format(savepath)))
        self.nel = int(self.js["system"]["electron"]["nelec"])
        self.npdep = self.js["input"]["wstat_control"]["n_pdep_eigen"]

        # PDEP eigenvalues
        if os.path.isfile(f"{savepath}/summary.json"):
            js = json.load(open("{}/summary.json".format(savepath)))
            self.evs = np.array(js["dielectric_matrix"]["pdep"][0]["eigenval"])
        else:
            self.evs = np.array(self.js["output"]["eigenval"])

        # fxc matrix
        try:
            self.fxc = self.parse_fxc("{}/FXC.dat".format(self.savepath))
        except:
            pass

    @staticmethod
    def parse_fxc(fxcfile):
        fxcxml = etree.iterparse(fxcfile, huge_tree=True)
        for event, leaf in fxcxml:
            if "fxc" in leaf.tag:
                fxc = np.fromstring(leaf.text, sep="\n")
                npdep = int(np.sqrt(len(fxc)))
                return fxc.reshape([npdep, npdep]).T
            leaf.clear()
        else:
            raise ValueError

    def make_renormalized_fxc(
        self,
        alpha=0.25,
        fxclim=-1,
        renormalization="cut",
        fxc_treatment="re-diagonalize",
        export=None,
    ):
        if fxc_treatment in ["re-diagonalize", "diagonal-only"]:
            fxcr = np.zeros([self.npdep, self.npdep])
        elif fxc_treatment in ["keep-offdiagonal"]:
            fxcr = self.fxc.copy()
        else:
            raise ValueError

        if fxc_treatment == "re-diagonalize":
            assert renormalization == "cut"
            ev, evc = np.linalg.eig((self.fxc + self.fxc.T) / 2)
            fxcr = evc @ np.diag(np.maximum(ev, fxclim)) @ evc.T
        else:
            for i in range(self.npdep):
                if renormalization == "erf":
                    fxcr[i, i] = (
                        erfc(alpha * i / self.nel) * self.fxc[i, i]
                        + erf(alpha * i / self.nel) * fxclim
                    )
                elif renormalization == "exp":
                    fxcr[i, i] = np.exp(-alpha * i / self.nel) * self.fxc[i, i]
                elif renormalization == "cut":
                    fxcr[i, i] = max(self.fxc[i, i], fxclim)
                elif renormalization == "none":
                    fxcr[i, i] = self.fxc[i, i]
                else:
                    raise ValueError

        if export is None:
            return fxcr
        else:
            fxcfile = open("{}/FXC.dat".format(self.savepath)).readlines()
            fxcfile[find_index("<fxc", fxcfile) + 1 : find_index("</fxc>", fxcfile)] = [
                "{:.15e}\n".format(fxcr[i, j])
                for i, j in np.ndindex(self.npdep, self.npdep)
            ]
            open("{}/{}".format(self.savepath, export), "w").writelines(fxcfile)

    @staticmethod
    def make_fxc_file(fxc, fname):
        npdep = fxc.shape[0]
        assert fxc.shape == (npdep, npdep)
        open(fname, "w").write(
            _fxc_template.format(
                npdep=npdep,
                fxcdim=fxc.size,
                fxcdata="\n".join(f"{x:.10e}" for x in fxc.flatten()),
            )
        )


class WfreqOutput:
    def __init__(self, outpath=None, savepath="."):
        if outpath is not None:
            ofile = open(outpath).readlines()
            self.nel = int(parse_one_value(float, ofile[find_index("nelec", ofile)]))
            self.nspin = parse_one_value(int, ofile[find_index("nspin", ofile)])
            self.qp1 = parse_many_values(
                2, int, ofile[find_index("qp_bandrange(1)", ofile)]
            )[1]
            self.qp2 = parse_many_values(
                2, int, ofile[find_index("qp_bandrange(2)", ofile)]
            )[1]
            iline = find_indices("K      B      QP energ. [eV] conv", ofile)[-1] + 2
            states = parse_many_values(self.qp2 - self.qp1 + 1, float, ofile[iline:])

        else:
            self.js = json.load(open("{}/wfreq.json".format(savepath)))
            if "output" not in self.js:
                raise ValueError("WfreqOutput: output section not found in wfreq.json.")

            self.nel = int(self.js["system"]["electron"]["nelec"])
            self.nspin = self.js["system"]["electron"]["nspin"]
            self.qp1, self.qp2 = self.js["input"]["wfreq_control"]["qp_bandrange"]
            states = self.js["output"]["Q"]["K000001"]["eqpSec"]

            self.walltime = self.js["timing"]["WFREQ"]["wall:sec"]

        if self.nspin != 1:
            raise NotImplementedError

        self.states = np.array(states)
        self.occ_states = self.states[0 : self.nel // 2 - self.qp1 + 1]
        self.vir_states = self.states[self.nel // 2 - self.qp1 + 1 :]

        try:
            self.homo = max(self.occ_states)
            # if self.homo != self.occ_states[-1]:
            #     print("WfreqOutput: HOMO@GW {} /= HOMO@DFT {}".format(self.homo, self.occ_states[-1]))
            self.lumo = min(self.vir_states)
            # if self.lumo != self.vir_states[0]:
            #     print("WfreqOutput: LUMO@GW {} /= LUMO@DFT {}".format(self.lumo, self.vir_states[0]))
            self.gap = self.lumo - self.homo
        except:
            print("WfreqOutput: error determining HOMO/LUMO")

    def plot_spectral_analysis(
        self,
        ikpt,
        ibnd,
        in_place=True,
        title="",
        plot_linear=False,
        plot_sigma=False,
        plot_width=2,
        find_maximum=False,
    ):
        """Spectral analysis of WFREQ spectral function calculation. See Govoni JCTC 2018 SI."""
        bandoffset = self.js["output"]["Q"]["bandmap"][0]
        ibnd -= bandoffset

        eks = self.js["output"]["Q"]["K{:06}".format(ikpt)]["eks"][ibnd]
        eqpsec = self.js["output"]["Q"]["K{:06}".format(ikpt)]["eqpSec"][ibnd]
        eqplin = self.js["output"]["Q"]["K{:06}".format(ikpt)]["eqpLin"][ibnd]
        sigmax = self.js["output"]["Q"]["K{:06}".format(ikpt)]["sigmax"][ibnd]
        sigmac_eks_re = self.js["output"]["Q"]["K{:06}".format(ikpt)]["sigmac_eks"][
            "re"
        ][ibnd]
        z = self.js["output"]["Q"]["K{:06}".format(ikpt)]["z"][ibnd]
        vxcl = self.js["output"]["Q"]["K{:06}".format(ikpt)]["vxcl"][ibnd]
        vxcnl = self.js["output"]["Q"]["K{:06}".format(ikpt)]["vxcnl"][ibnd]
        vxc = vxcl + vxcnl

        omega = np.array(self.js["output"]["P"]["freqlist"])
        sigmac_re = np.array(
            self.js["output"]["P"]["K{:06}".format(ikpt)]["B{:06}".format(ibnd + 1)][
                "sigmac"
            ]["re"]
        )
        sigmac_im = np.array(
            self.js["output"]["P"]["K{:06}".format(ikpt)]["B{:06}".format(ibnd + 1)][
                "sigmac"
            ]["im"]
        )
        sigma = sigmax + sigmac_re + 1j * sigmac_im

        # Equation 1 of Govoni 2018 JCTC SI
        Q = sigma.real - vxc + eks - omega
        # Equation 2 of Govoni 2018 JCTC SI
        Qlin = sigmax + sigmac_eks_re - (1 / z) * (omega - eks) - vxc
        # Equation 3 of Govoni 2018 JCTC SI
        A = (
            (1 / np.pi)
            * np.abs(sigma.imag)
            / ((omega - eks - sigma.real + vxc) ** 2 + (sigma.imag) ** 2)
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        fig.set_dpi(300)

        quantities_info = {
            "A": {"x": omega, "y": A, "label": "$A$", "color": "blue"},
            "Q": {"x": omega, "y": Q, "label": "$Q$", "color": "green"},
            "Qlin": {
                "x": omega,
                "y": Qlin,
                "label": "$Q^{\mathrm{lin}}$",
                "color": "red",
            },
            "sigmac_re": {
                "x": omega,
                "y": sigmac_re,
                "label": "$Re(\Sigma_c)$",
                "color": "purple",
            },
            "sigmac_im": {
                "x": omega,
                "y": sigmac_im,
                "label": "$Im(\Sigma_c)$",
                "color": "brown",
            },
        }

        quantities_to_plot = ["A", "Q"]

        if plot_linear:
            quantities_to_plot += ["Qlin"]
        if plot_sigma:
            quantities_to_plot += ["sigmac_re", "sigmac_im"]

        ys = []
        for quantity in quantities_to_plot:
            x = quantities_info[quantity].pop("x")
            y = quantities_info[quantity].pop("y")
            ax.plot(x, y, **quantities_info[quantity])
            ys.append(y)

        if find_maximum:
            print("argmax(A) = {}, eqpsec = {}".format(omega[np.argmax(A)], eqpsec))
            eqpsec = omega[np.argmax(A)]

        imin = np.argmax(omega > eqpsec - plot_width)
        imax = np.argmin(omega < eqpsec + plot_width)
        ax.set_xlim(omega[imin], omega[imax])
        ax.set_ylim(
            min(min(y[imin:imax]) for y in ys), max(max(y[imin:imax]) for y in ys)
        )

        auxlines_info = {
            "zero": {"x": ax.get_xlim(), "y": [0, 0], "color": "black"},
            "eqpsec": {
                "x": [eqpsec, eqpsec],
                "y": ax.get_ylim(),
                "label": "$E^{\mathrm{QP}}$",
                "linestyle": "--",
                "color": "green",
            },
            "eqplin": {
                "x": [eqplin, eqplin],
                "y": ax.get_ylim(),
                "label": "$E^{\mathrm{QP-lin}}$",
                "linestyle": "--",
                "color": "red",
            },
            "sigmax": {
                "x": ax.get_xlim(),
                "y": [sigmax, sigmax],
                "label": "$\Sigma_x$",
                "color": "yellow",
            },
        }

        auxlines_to_plot = ["eqpsec"]
        if plot_linear:
            auxlines_to_plot += ["eqplin"]
        # if plot_sigma:
        #     auxlines_to_plot += ["sigmax"]

        for auxline in auxlines_to_plot:
            x = auxlines_info[auxline].pop("x")
            y = auxlines_info[auxline].pop("y")
            ax.plot(x, y, **auxlines_info[auxline])

        label_order = [
            "$A$",
            "$Q$",
            "$E^{\mathrm{QP}}$",
            "$Q^{\mathrm{lin}}$",
            "$E^{\mathrm{QP-lin}}$",
            "$\Sigma_x$",
            "$Re(\Sigma_c)$",
            "$Im(\Sigma_c)$",
        ]
        handles, labels = ax.get_legend_handles_labels()
        hls = sorted(zip(handles, labels), key=lambda hl: label_order.index(hl[1]))
        ax.legend(*zip(*hls))

        ax.set_xlabel("$\omega$ (eV)")
        ax.set_ylabel("Energy (eV)")
        ax.set_title(title)

        if in_place:
            return
        else:
            return fig, ax
