import base64
import h5py
import numpy as np
from xml.etree import ElementTree as ET


class Qbox2BSE(object):
    """Parses Qbox output and generates files needed by WEST BSE.

    Args:
        filename (string): Qbox output file (XML)

    :Example:

    >>> from westpy.bse import *
    >>> qb = Qbox2BSE("qb.out")
    """

    def __init__(self, filename: str):
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

    def write_localization(self, filename: str = "bis_info"):
        """
        Reads localization from XML file then writes to text file.

        Args:
            filename (string): name of Qbox bisection information file

        :Example:

        >>> from westpy.bse import *
        >>> qb = Qbox2BSE("qb.out")
        >>> qb.write_localization()
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

    def write_wavefunction(self, filename: str = "qb_wfc"):
        """
        Reads wavefunctions from XML file then writes to HDF5 file.

        Args:
            filename (string): name of Qbox wavefunction file

        :Example:

        >>> from westpy.bse import *
        >>> qb = Qbox2BSE("qb.out")
        >>> qb.write_wavefunction()
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

            nwfc = self.nwfc[ispin]
            nx = self.ngrid[0]
            ny = self.ngrid[1]
            nz = self.ngrid[2]

            with h5py.File(thisname, "w") as f:
                wfcs = f.create_group("wfcs")
                wfcs.attrs.create("nwfcs", nwfc)
                wfcs.attrs.create("nx", nx)
                wfcs.attrs.create("ny", ny)
                wfcs.attrs.create("nz", nz)

                for igf, gf in enumerate(gfs):
                    # get base64 string without line breaks
                    s = gf.text.replace("\n", "")

                    # base64 -> bytes
                    b = base64.b64decode(s)

                    # bytes -> numpy
                    array = np.frombuffer(b, dtype="float64")

                    wfcs.create_dataset(f"wfc{igf+1}", data=array, compression="gzip")
