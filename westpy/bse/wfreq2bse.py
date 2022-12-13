import json
from westpy.units import eV


class Wfreq2BSE(object):
    def __init__(self, filename: str):
        """Parses Wfreq output and generates quasiparticle energy file needed by WEST BSE.

        :param filename: Wfreq output file (JSON)
        :type filename: string

        :Example:

        >>> from westpy.bse import *
        >>> qp = Wfreq2BSE("wfreq.json")
        """

        self.filename = filename

    def write_qp_correction(self, filename: str = "qp_eig"):
        """
        Reads quasiparticle energies from JSON file then writes to text file.

        :param filename: name of quasiparticle energy file
        :type filename: string

        :Example:

        >>> from westpy.bse import *
        >>> qp = Wfreq2BSE("qb.out")
        >>> qp.write_qp_correction()
        """

        with open(self.filename, "r") as f:
            res = json.load(f)

        nspin = res["system"]["electron"]["nspin"]

        for ispin in range(nspin):
            thisname = f"{filename}.{ispin+1}"

            qps = res["output"]["Q"][f"K{ispin+1:0>6}"]["eqpSec"]
            nqp = len(qps)

            with open(thisname, "w") as f:
                f.write(f"{nqp}\n")

                for qp in qps:
                    f.write(f"{qp*eV}\n")
