from typing import List
import json
import numpy as np
import scipy.constants as sc
import scipy.linalg.lapack as la
from westpy.units import eV


class BSEResult(object):
    def __init__(self, filename: str):
        """Parses Wbse Lanczos output and plots absorption spectrum.

        :param filename: Wbse output file (JSON)
        :type filename: string

        :Example:

        >>> from westpy.bse import *
        >>> wbse = BSEResult("wbse.json")
        """

        self.filename = filename

        with open(filename, "r") as f:
            res = json.load(f)

        self.nspin = res["system"]["electron"]["nspin"]
        which = res["input"]["wbse_control"]["wbse_calculation"]
        assert which in ["L", "l"]
        self.n_lanczos = res["input"]["wbse_control"]["n_lanczos"]
        pol = res["input"]["wbse_control"]["ipol_input"]
        if pol in ["XYZ", "xyz"]:
            self.n_ipol = 3
            self.pols = ["XX", "YY", "ZZ"]
            self.can_do = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY", "ZZ"]
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

    def poltSpectrum(
        self,
        ipol: str = None,
        ispin: int = 1,
        energyRange: List[float] = [0.0, 10.0, 0.01],
        sigma: float = 0.1,
        n_extra: int = 0,
        fname: str = None,
        save_data: bool = True,
    ) -> None:
        """Parses Wbse Lanczos output and plots absorption spectrum.

        :param ispin: Spin channel to consider
        :type ispin: int
        :param energyRange: energy range = min, max, step (eV)
        :type energyRange: 3-dim float
        :param sigma: Broadening width (eV)
        :type sigma: float
        :param n_extra: Number of extrapolation steps
        :type n_extra: int
        :param fname: Output file name
        :type fname: string

        :Example:

        >>> from westpy.bse import *
        >>> wbse = BSEResult("wbse.json")
        >>> wbse.plotSpectrum(energyRange=[0.0,10.0,0.01],sigma=0.1,n_extra=100000)
        """

        assert ipol in self.can_do
        assert ispin >= 1 and ispin <= self.nspin
        xmin, xmax, dx = energyRange
        assert xmax > xmin
        assert dx > 0.0
        assert sigma > 0.0
        assert n_extra >= 0

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

        if self.n_ipol == 1:
            ip = 0
            ip2 = 0
        elif self.n_ipol == 3:
            if ipol[0] == "X":
                ip = 0
            if ipol[0] == "Y":
                ip = 1
            if ipol[0] == "Z":
                ip = 2
            if ipol[1] == "X":
                ip2 = 0
            if ipol[1] == "Y":
                ip2 = 1
            if ipol[1] == "Z":
                ip2 = 2

        raw = open(f"chi_{ipol}.dat", "w")
        raw.write(f"chi_{ipol} \hbar \omega (eV) real(chi) (e^2*a_0^2/eV) imag(chi) (e^2*a_0^2/eV)")

        n_step = int((xmax - xmin) / dx) + 1
        for i_step in range(n_step):
            # eV to Ry
            freq_ev = xmin * eV
            sigma_ev = sigma * eV

            # calculate susceptibility for given frequency
            chi = self.__calc_chi(freq_ev, sigma_ev)

            # 1/Ry to 1/eV
            chi = chi * eV

            raw.write(
                f"chi_{ipol} {xmin:15.8e} {chi[ip2, ip].real:15.8e} {chi[ip2, ip].imag:15.8e}\n"
            )

            xmin = xmin + dx

        raw.close()

    def __read_beta_zeta(self, ispin: int):

        self.norm = np.zeros(self.n_ipol, dtype=np.float64)
        self.beta = np.zeros((self.n_ipol, self.n_total), dtype=np.float64)
        self.zeta = np.zeros((self.n_ipol, 3, self.n_total), dtype=np.complex128)

        with open(self.filename, "r") as f:
            res = json.load(f)

        for ip, lp in enumerate(self.pols):
            if self.nspin > 1:
                tmp = res["output"]["lanczos"][f"K{ispin:6.6d}"][lp]["beta"]
            else:
                tmp = res["output"]["lanczos"][lp]["beta"]

            beta_read = np.array(tmp)

            if self.nspin > 1:
                tmp = res["output"]["lanczos"][f"K{ispin:6.6d}"][lp]["zeta"]
            else:
                tmp = res["output"]["lanczos"][lp]["zeta"]

            zeta_read = np.array(tmp).reshape((3, self.n_lanczos))

            self.norm[ip] = beta_read[0]
            self.beta[ip, 0 : self.n_lanczos - 1] = beta_read[1 : self.n_lanczos]
            self.beta[ip, self.n_lanczos - 1] = beta_read[self.n_lanczos - 1]
            self.zeta[ip, :, 0 : self.n_lanczos] = zeta_read[:, 0 : self.n_lanczos]

    def __extrapolate(self, n_extra: int):

        skip = False

        if n_extra > 0:
            average = np.zeros(self.n_ipol, dtype=np.float64)
            amplitude = np.zeros(self.n_ipol, dtype=np.float64)

            for ip in range(self.n_ipol):
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

        return chi
