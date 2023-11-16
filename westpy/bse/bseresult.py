from typing import List
import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg.lapack as la
from westpy.units import eV


class BSEResult(object):
    def __init__(self, filename: str):
        """Parses BSE/TDDFT results.

        :param filename: Wbse or Westpp output file (JSON)
        :type filename: string

        :Example:

        >>> from westpy.bse import *
        >>> wbse = BSEResult("wbse.json")
        """

        self.filename = filename

        with open(filename, "r") as f:
            res = json.load(f)

        if res["software"]["program"] == "WBSE":
            self.wbse_calc = "lanczos"
        elif res["software"]["program"] == "WESTPP":
            self.wbse_calc = "davidson"
        assert self.wbse_calc is not None

        if self.wbse_calc == "lanczos":
            self.nspin = res["system"]["electron"]["nspin"]
            which = res["input"]["wbse_control"]["wbse_calculation"]
            assert which in ["L", "l"]
            self.n_lanczos = res["input"]["wbse_control"]["n_lanczos"]
            pol = res["input"]["wbse_control"]["wbse_ipol"]
            if pol in ["XYZ", "xyz"]:
                self.n_ipol = 3
                self.pols = ["XX", "YY", "ZZ"]
                self.can_do = [
                    "XX",
                    "XY",
                    "XZ",
                    "YX",
                    "YY",
                    "YZ",
                    "ZX",
                    "ZY",
                    "ZZ",
                    "XYZ",
                ]
            elif pol in ["XX", "xx"]:
                self.n_ipol = 1
                self.pols = ["XX"]
                self.can_do = ["XX"]
            elif pol in ["YY", "yy"]:
                self.n_ipol = 1
                self.pols = ["YY"]
                self.can_do = ["YY"]
            elif pol in ["ZZ", "zz"]:
                self.n_ipol = 1
                self.pols = ["ZZ"]
                self.can_do = ["ZZ"]
            self.dip_real = res["input"]["wbse_control"]["l_dipole_realspace"]
            self.bg = np.zeros((3, 3), dtype=np.float64)
            self.bg[0] = np.array(res["system"]["cell"]["b1"])
            self.bg[1] = np.array(res["system"]["cell"]["b2"])
            self.bg[2] = np.array(res["system"]["cell"]["b3"])
            self.bg = self.bg / res["system"]["cell"]["tpiba"]

        elif self.wbse_calc == "davidson":
            self.nspin = res["system"]["electron"]["nspin"]
            self.n_liouville = res["input"]["westpp_control"][
                "westpp_n_liouville_to_use"
            ]
            self.n_ipol = 3
            self.pols = ["XX", "YY", "ZZ"]
            self.can_do = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ", "XYZ"]

    def plotSpectrum(
        self,
        ipol: str = None,
        ispin: int = 1,
        energyRange: List[float] = [0.0, 10.0, 0.01],
        sigma: float = 0.1,
        n_extra: int = 0,
        fname: str = None,
    ):
        """Plots BSE/TDDFT absorption spectrum.

        :param ipol: which component to compute ("XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ", or "XYZ")
        :type ipol: string
        :param ispin: Spin channel to consider
        :type ispin: int
        :param energyRange: energy range = min, max, step (eV)
        :type energyRange: 3-dim float
        :param sigma: Broadening width (eV)
        :type sigma: float
        :param n_extra: Number of extrapolation steps (Lanczos only)
        :type n_extra: int
        :param fname: Output file name
        :type fname: string

        :Example:

        >>> from westpy.bse import *
        >>> wbse = BSEResult("wbse.json")
        >>> wbse.plotSpectrum(ipol="XYZ",energyRange=[0.0,10.0,0.01],sigma=0.1,n_extra=100000)
        """

        assert ipol in self.can_do
        assert ispin >= 1 and ispin <= self.nspin
        xmin, xmax, dx = energyRange
        assert xmax > xmin
        assert dx > 0.0
        assert sigma > 0.0
        assert n_extra >= 0

        if self.wbse_calc == "lanczos":
            if self.n_lanczos < 151 and n_extra > 0:
                n_extra = 0
            self.n_total = self.n_lanczos + n_extra

            self.__read_beta_zeta(ispin)
            self.__extrapolate(n_extra)

            self.r = np.zeros(self.n_total, dtype=np.complex128)
            self.r[0] = 1.0

            self.b = np.zeros((self.n_ipol, self.n_total - 1), dtype=np.complex128)
            for ip in range(self.n_ipol):
                for i in range(self.n_total - 1):
                    self.b[ip, i] = -self.beta[ip, i]
        elif self.wbse_calc == "davidson":
            self.__read_tdm()

        sigma_ev = sigma * eV
        n_step = int((xmax - xmin) / dx) + 1
        energyAxis = np.linspace(xmin, xmax, n_step, endpoint=True)
        chiAxis = np.zeros(n_step, dtype=np.complex128)

        for ie, energy in enumerate(energyAxis):
            # eV to Ry
            freq_ev = energy * eV

            # calculate susceptibility for given frequency
            chi = self.__calc_chi(freq_ev, sigma_ev)

            # 1/Ry to 1/eV
            chi = chi * eV

            if self.n_ipol == 1:
                chiAxis[ie] = chi[0, 0]
            elif self.n_ipol == 3:
                # crystal to cart
                if self.wbse_calc == "lanczos" and self.dip_real == False:
                    chi = np.dot(self.bg.T, np.dot(chi, self.bg))

                if ipol == "XX":
                    chiAxis[ie] = chi[0, 0]
                if ipol == "XY":
                    chiAxis[ie] = chi[1, 0]
                if ipol == "XZ":
                    chiAxis[ie] = chi[2, 0]
                if ipol == "YX":
                    chiAxis[ie] = chi[0, 1]
                if ipol == "YY":
                    chiAxis[ie] = chi[1, 1]
                if ipol == "YZ":
                    chiAxis[ie] = chi[2, 1]
                if ipol == "ZX":
                    chiAxis[ie] = chi[0, 2]
                if ipol == "ZY":
                    chiAxis[ie] = chi[1, 2]
                if ipol == "ZZ":
                    chiAxis[ie] = chi[2, 2]
                if ipol == "XYZ":
                    chiAxis[ie] = (
                        (chi[0, 0] + chi[1, 1] + chi[2, 2]) * energy / 3.0 / np.pi
                    )

        print(f"plotting absorption spectrum ({self.wbse_calc.capitalize()})")

        if not fname:
            fname = f"chi_{ipol}.png"

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        dosPlot = ax.plot(energyAxis, chiAxis.imag, label=f"chi_{ipol}")

        plt.xlim([xmin, xmax])
        plt.xlabel("$\omega$ (eV)")
        if ipol == "XYZ":
            plt.ylabel("abs. coeff. (a.u.)")
        else:
            plt.ylabel("Im[$\chi$] (a.u.)")
            plt.legend()
        plt.savefig(fname, dpi=300)
        print("output written in : ", fname)
        print("waiting for user to close image preview...")
        plt.show()
        fig.clear()

    def __read_beta_zeta(self, ispin: int):
        self.norm = np.zeros(self.n_ipol, dtype=np.float64)
        self.beta = np.zeros((self.n_ipol, self.n_total), dtype=np.float64)
        self.zeta = np.zeros((self.n_ipol, 3, self.n_total), dtype=np.complex128)

        with open(self.filename, "r") as f:
            res = json.load(f)

        for ip, lp in enumerate(self.pols):
            tmp = res["output"]["lanczos"][f"K{ispin:0>6}"][lp]["beta"]
            beta_read = np.array(tmp)

            tmp = res["output"]["lanczos"][f"K{ispin:0>6}"][lp]["zeta"]
            zeta_read = np.array(tmp).reshape((3, self.n_lanczos))

            self.norm[ip] = beta_read[0]
            self.beta[ip, 0 : self.n_lanczos - 1] = beta_read[1 : self.n_lanczos]
            self.beta[ip, self.n_lanczos - 1] = beta_read[self.n_lanczos - 1]
            self.zeta[ip, :, 0 : self.n_lanczos] = zeta_read[:, 0 : self.n_lanczos]

    def __extrapolate(self, n_extra: int):
        if n_extra > 0:
            average = np.zeros(self.n_ipol, dtype=np.float64)
            amplitude = np.zeros(self.n_ipol, dtype=np.float64)

            for ip in range(self.n_ipol):
                skip = False
                counter = 0

                for i in range(150, self.n_lanczos):
                    if skip:
                        skip = False
                        continue

                    if i % 2 == 0:
                        if (
                            i != 150
                            and abs(self.beta[ip, i] - average[ip] / counter) > 2.0
                        ):
                            skip = True
                        else:
                            average[ip] = average[ip] + self.beta[ip, i]
                            amplitude[ip] = amplitude[ip] + self.beta[ip, i]
                            counter = counter + 1
                    else:
                        if (
                            i != 150
                            and abs(self.beta[ip, i] - average[ip] / counter) > 2.0
                        ):
                            skip = True
                        else:
                            average[ip] = average[ip] + self.beta[ip, i]
                            amplitude[ip] = amplitude[ip] - self.beta[ip, i]
                            counter = counter + 1

                average[ip] = average[ip] / counter
                amplitude[ip] = amplitude[ip] / counter

            for ip in range(self.n_ipol):
                for i in range(self.n_lanczos - 1, self.n_total):
                    if i % 2 == 0:
                        self.beta[ip, i] = average[ip] + amplitude[ip]
                    else:
                        self.beta[ip, i] = average[ip] - amplitude[ip]

    def __calc_chi(self, freq: float, broaden: float):
        degspin = 2.0 / self.nspin
        omeg_c = freq + broaden * 1j

        chi = np.zeros((self.n_ipol, self.n_ipol), dtype=np.complex128)

        if self.wbse_calc == "lanczos":
            for ip2 in range(self.n_ipol):
                a = np.full(self.n_total, omeg_c, dtype=np.complex128)
                b = self.b[ip2, :]
                c = b
                r = self.r

                b1, a1, c1, r1, ierr = la.zgtsv(b, a, c, r)
                assert ierr == 0

                for ip in range(self.n_ipol):
                    chi[ip2, ip] = np.dot(self.zeta[ip2, ip, :], r1)
                    chi[ip2, ip] *= -2.0 * degspin * self.norm[ip2]

        elif self.wbse_calc == "davidson":
            for ip in range(self.n_ipol):
                for ip2 in range(self.n_ipol):
                    num = self.tdm[:, ip] * self.tdm[:, ip2]
                    den = freq - self.vee[:] - 1j * broaden
                    tmp = np.sum(num / den)
                    chi[ip, ip2] = tmp

        return chi

    def __read_tdm(self):
        self.tdm = np.zeros((self.n_liouville, 3), dtype=np.float64)
        self.vee = np.zeros(self.n_liouville, dtype=np.float64)

        with open(self.filename, "r") as f:
            res = json.load(f)

        for il in range(self.n_liouville):
            tmp = res["output"][f"E{(il+1):05d}"]["transition_dipole_moment"]
            self.tdm[il] = np.array(tmp)
            tmp = res["output"][f"E{(il+1):05d}"]["excitation_energy"]
            self.vee[il] = tmp
