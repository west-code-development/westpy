import base64
from xml.etree import ElementTree as ET


class Qbox2BSE(object):
    def __init__(self, filename: str):
        """Parses Qbox output and generates files needed by WEST BSE.

        :param filename: Qbox output file (XML)
        :type filename: string

        :Example:

        >>> from westpy import *
        >>> qb2bse = Qbox2BSE("qb.out")
        """

        self.filename = filename

        root = ET.parse(filename)
        wfc = root.find("wavefunction")
        self.nspin = int(wfc.attrib["nspin"])
        grid = wfc.find("grid")
        self.ngrid = [
            int(grid.attrib["nx"]),
            int(grid.attrib["ny"]),
            int(grid.attrib["nz"]),
        ]
        sds = wfc.findall("slater_determinant")
        self.nwfc = []
        for sd in sds:
            self.nwfc.append(int(sd.attrib["size"]))

    def write_localization(self, filename: str = "info.bis"):
        """
        Reads localization from XML file then writes to text file.

        :param filename: name of Qbox bisection information file
        :type filename: string

        :Example:

        >>> from westpy import *
        >>> qb2bse = Qbox2BSE("qb.out")
        >>> qb2bse.write_localization()
        """

        localization = {}

        with open(self.filename, "r") as f:
            lines = f.readlines()

            ispin = -1

            for line in lines:
                if line.strip().startswith("BisectionCmd"):
                    ispin += 1
                    localization[ispin] = []

                if line.strip().startswith("localization"):
                    localization[ispin].append(line.split()[1])

        for ispin in range(self.nspin):
            thisname = f"{filename}.{ispin+1}"

            with open(thisname, "w") as f:
                f.write(f"{self.nwfc[ispin]}\n")

                for loc in localization[ispin]:
                    f.write(f"{loc}\n")

    def write_wavefunction(self, filename: str = "qb.wfc"):
        """
        Reads wavefunctions from XML file then writes to binary file.

        :param filename: name of Qbox wavefunction file
        :type filename: string

        :Example:

        >>> from westpy import *
        >>> qb2bse = Qbox2BSE("qb.out")
        >>> qb2bse.write_wavefunction()
        """

        with open(self.filename, "r") as f:
            lines = f.readlines()

            for line in lines:
                if line.strip().startswith("[qbox] <cmd>save"):
                    # get file name without </cmd>
                    bis_filename = line.split()[2][:-6]
                    break

        root = ET.parse(bis_filename)

        wavefunction = {}

        wfc = root.find("wavefunction")
        sds = wfc.findall("slater_determinant")

        for ispin, sd in enumerate(sds):
            thisname = f"{filename}.{ispin+1}"
            gfs = sd.findall("grid_function")

            with open(thisname, "wb") as f:
                f.write(self.nwfc[ispin].to_bytes(4, "little"))
                f.write(self.ngrid[0].to_bytes(4, "little"))
                f.write(self.ngrid[1].to_bytes(4, "little"))
                f.write(self.ngrid[2].to_bytes(4, "little"))

                for gf in gfs:
                    # get base64 string without line breaks
                    s = gf.text.replace("\n", "")

                    # base64 -> bytes
                    b = base64.b64decode(s)

                    # write bytes
                    f.write(b)
