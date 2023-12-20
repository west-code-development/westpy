class Geometry(object):
    """Class for representing a set of atoms in a periodic cell.

    :Example:

    >>> from westpy import *
    >>> geom = Geometry()
    >>> geom.setCell( (1,0,0), (0,1,0), (0,0,1) )
    >>> geom.addAtom( symbol="Si", abs_coord=(0,0,0) )
    >>> geom.addSpecies( "Si", "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf" )

    .. note:: Vectors are set in a.u. by default. If you set units=Angstrom a coversion to a.u. will be made.

    """

    #
    from westpy import Bohr

    #
    def __init__(self, cell=None):
        self.cell = {}
        self.atoms = []
        self.species = {}
        self.isSet = {}
        self.isSet["cell"] = False
        self.isSet["atoms"] = False
        self.isSet["species"] = False
        self.pseudoFormat = None

    #
    def __atomsMatchSpecies(self):
        """Checks if atoms match species"""
        matches = self.isSet["species"] and self.isSet["atoms"]
        for atom in self.atoms:
            matches = matches and atom.symbol in self.species.keys()
        return matches

    #
    def isValid(self):
        """Checks if geometry is valid

        The method checks that:
           - the cell is set
           - at least one atom has been added
           - the pseudopotentials of all species are defined
           - the pseudopotentials do not contain a mix of upf and xml formats
        """
        isValid = True
        for key in self.isSet.keys():
            isValid = isValid and self.isSet[key]
            if not self.isSet[key]:
                print("ERR: set " + key)
        isValid = isValid and self.__atomsMatchSpecies()
        isValid = isValid and self.pseudoFormat in ["upf", "xml"]
        return isValid

    #
    def addSpecies(self, symbol, url):
        """Adds a species.

        :param symbol: chemical symbol
        :type symbol: string
        :param url: url
        :type url: string

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addSpecies( "Si", "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf" )

        .. note:: You can use this method to add either upf or xml pseudopotentials. However it is forbidded to mix them.
        """
        from westpy import extractFileNamefromUrl

        fname = extractFileNamefromUrl(url)
        this_pseudo_format = None
        if fname.endswith("upf") or fname.endswith("UPF"):
            this_pseudo_format = "upf"
        if fname.endswith("xml") or fname.endswith("XML"):
            this_pseudo_format = "xml"
        assert this_pseudo_format in ["upf", "xml"]
        from mendeleev import element

        el = element(symbol)
        self.species[symbol] = {}
        self.species[symbol]["fname"] = fname
        self.species[symbol]["url"] = url
        self.species[symbol]["symbol"] = el.symbol
        self.species[symbol]["atomic_number"] = el.atomic_number
        self.species[symbol]["name"] = el.name
        self.species[symbol]["mass"] = el.mass
        self.species[symbol]["format"] = this_pseudo_format
        self.__updatePseudoFormat()
        self.isSet["species"] = True

    #
    def __updatePseudoFormat(self):
        """pseudo format is either upf, xml or mixed."""
        f = []
        for key in self.species.keys():
            f.append(self.species[key]["format"])
        self.pseudoFormat = "mixed"
        if all(form == "upf" for form in f):
            self.pseudoFormat = "upf"
        if all(form == "xml" for form in f):
            self.pseudoFormat = "xml"

    #
    def setCell(self, a1=(0, 0, 0), a2=(0, 0, 0), a3=(0, 0, 0), units=Bohr):
        """Sets cell, given the three vectors :math:`a_1`, :math:`a_2`, :math:`a_3`.

        :param a1: :math:`a_1`
        :type a1: 3-dim tuple
        :param a2: :math:`a_2`
        :type a2: 3-dim tuple
        :param a3: :math:`a_3`
        :type a3: 3-dim tuple
        :param units: Units, optional
        :type units: "Bohr" or "Angstrom"

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.setCell( (1,0,0), (0,1,0), (0,0,1) )
        """
        import numpy as np

        #
        self.cell["a1"] = np.array(a1, dtype=np.float64) * units
        self.cell["a2"] = np.array(a2, dtype=np.float64) * units
        self.cell["a3"] = np.array(a3, dtype=np.float64) * units
        self.cell["volume"] = np.dot(
            self.cell["a1"], np.cross(self.cell["a2"], self.cell["a3"])
        )
        self.cell["b1"] = (
            2.0
            * np.pi
            * np.cross(self.cell["a2"], self.cell["a3"])
            / self.cell["volume"]
        )
        self.cell["b2"] = (
            2.0
            * np.pi
            * np.cross(self.cell["a3"], self.cell["a1"])
            / self.cell["volume"]
        )
        self.cell["b3"] = (
            2.0
            * np.pi
            * np.cross(self.cell["a1"], self.cell["a2"])
            / self.cell["volume"]
        )
        self.isSet["cell"] = True

    #
    def addAtom(self, symbol, position, units=Bohr):
        """Adds a single atom.

        :param symbol: chemical symbol
        :type symbol: string
        :param position: position
        :type position: 3-dim tuple
        :param units: Units, optional
        :type units: "Bohr" or "Angstrom"

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addAtom( position="Si", abs_coord=(0,0,0) )
        """
        from westpy import Atom

        self.atoms.append(Atom(symbol=symbol, position=np.array(position) * units))
        self.isSet["atoms"] = True

    #
    def addFracCoordAtom(self, symbol, frac_coord):
        """adds a single atom by fractional coords
        :param symbol: chemical symbol
        :type symbol: string
        :param position: position
        :type position: 3-dim tuple

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addFracCoordAtom( "Si", (0,1/3.0,2/3.0) )
        """
        if not self.isSet["cell"]:
            print("Set cell first!")
            return
        import numpy as np
        from westpy import Atom, Bohr

        self.atoms.append(
            Atom(
                symbol=symbol,
                abs_coord=np.asarray(
                    frac_coord[0] * self.cell["a1"]
                    + frac_coord[1] * self.cell["a2"]
                    + frac_coord[2] * self.cell["a3"]
                )
                * Bohr,
            )
        )
        self.isSet["atoms"] = True

    def __addAtomsFromXYZLines(self, lines, decode=True):
        """Adds atoms from XYZ lines.

        :param lines: lines read from XYZ file (only one image)
        :type lines: list of string
        :param decode:
        :type bool:
        """
        #
        from westpy import Angstrom

        natoms = int(lines[0])
        for line in lines[2 : 2 + natoms]:
            symbol, x, y, z = line.split()[:4]
            if decode:
                self.addAtom(
                    symbol=symbol.decode("utf-8"),
                    abs_coord=np.array([float(x), float(y), float(z)]) * Angstrom,
                )
            else:
                self.addAtom(
                    symbol=symbol,
                    abs_coord=np.array([float(x), float(y), float(z)]) * Angstrom,
                )

    #
    def addAtomsFromXYZFile(self, fname):
        """Adds atoms from XYZ file (only one image).

        :param fname: file name
        :type fname: string

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addAtomFromXYZFile( "CH4.xyz" )
        """
        #
        with open(fname, "r") as file:
            lines = file.readlines()
        self.__addAtomsFromXYZLines(lines, decode=False)

    #
    def addAtomsFromOnlineXYZ(self, url):
        """Adds atoms from XYZ file (only one image) located at url.

        :param url: url
        :type url: string

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addAtomsFromOnlineXYZ( "https://west-code.org/database/gw100/xyz/CH4.xyz" )
        """
        #
        import urllib.request

        with urllib.request.urlopen(url) as response:
            lines = response.readlines()
        self.__addAtomsFromXYZLines(lines, decode=True)

    #
    def getNumberOfAtoms(self):
        """Returns number of atoms.

        :returns: number of atoms
        :rtype: int

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addAtomsFromOnlineXYZ( "https://west-code.org/database/gw100/xyz/CH4.xyz" )
        >>> nat = geom.getNumberOfAtoms()
        >>> print( nat )
        5
        """
        nat = len(self.atoms)
        return nat

    #
    def getNumberOfSpecies(self):
        """Returns number of species.

        :returns: number of species
        :rtype: int

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addAtomsFromOnlineXYZ( "https://west-code.org/database/gw100/xyz/CH4.xyz" )
        >>> ntyp = geom.getNumberOfSpecies()
        >>> print( ntyp )
        2
        """
        sp = []
        for atom in self.atoms:
            if atom.symbol not in sp:
                sp.append(atom.symbol)
        ntyp = len(sp)
        return ntyp

    #
    def getNumberOfElectrons(self):
        """Returns number of electrons.

        :returns: number of electrons
        :rtype: int

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addAtomsFromOnlineXYZ( "https://west-code.org/database/gw100/xyz/CH4.xyz" )
        >>> geom.addSpecies( "C", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/C_ONCV_PBE-1.0.xml")
        >>> geom.addSpecies( "H", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/H_ONCV_PBE-1.0.xml")
        >>> nelec = geom.getNumberOfElectrons()
        >>> print( nelec )
        8
        """
        assert self.__atomsMatchSpecies()
        #
        nelec = 0
        if self.pseudoFormat in ["upf"]:
            from westpy import listLinesWithKeyfromOnlineText

            for atom in self.atoms:
                symbol = atom.symbol
                url = self.species[symbol]["url"]
                resp = listLinesWithKeyfromOnlineText(url, "z_valence")[0]
                this_valence = float(resp.decode("utf-8").split('"')[1])
                nelec += this_valence
        if self.pseudoFormat in ["xml"]:
            from westpy import listValuesWithKeyFromOnlineXML

            for atom in self.atoms:
                symbol = atom.symbol
                url = self.species[symbol]["url"]
                this_valence = float(
                    listValuesWithKeyFromOnlineXML(url, "valence_charge")[0]
                )
                nelec += this_valence
        return int(nelec)

    #
    def downloadPseudopotentials(self):
        """Download Pseudopotentials.

        :Example:

        >>> from westpy import *
        >>> geom = Geometry()
        >>> geom.addAtomsFromOnlineXYZ( "https://west-code.org/database/gw100/xyz/CH4.xyz" )
        >>> geom.addSpecies( "C", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/C_ONCV_PBE-1.0.xml")
        >>> geom.addSpecies( "H", "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/H_ONCV_PBE-1.0.xml")
        >>> geom.downloadPseudopotentials()

        .. note:: Pseudopotential files will be downloaded in the current directory.
        """
        assert self.__atomsMatchSpecies()
        #
        from westpy import download

        for key in self.species.keys():
            download(self.species[key]["url"], fname=self.species[key]["fname"])

    def view(self, style="stick", width=800, height=800, ix=1, iy=1, iz=1, debug=False):
        """Display simulation box geom in Angstrom.
        ix, iy, iz is the perodic display to system
        style can be line, stick, sphere.

        :param style:
        :param width:
        :param height:
        :param ix:
        :param iy:
        :param iz:
        :param debug:
        :return:
        """
        import py3Dmol
        import numpy as np
        from westpy.units import Angstrom

        BOHR2A = 1.0 / Angstrom
        a1 = self.cell["a1"] * BOHR2A
        a2 = self.cell["a2"] * BOHR2A
        a3 = self.cell["a3"] * BOHR2A
        nat = self.getNumberOfAtoms()
        times = ix * iy * iz
        if times > 1:
            nat *= times
        # generate xyz data
        xyz = str(nat) + "\n\n"
        for atom in self.atoms:
            if times:
                for i in range(ix):
                    for j in range(iy):
                        for k in range(iz):
                            xyz += (
                                atom.symbol
                                + " "
                                + " ".join(
                                    map(
                                        str,
                                        atom.position * BOHR2A
                                        + i * a1
                                        + j * a2
                                        + k * a3,
                                    )
                                )
                                + "\n"
                            )
        # creat viewer
        xyzview = py3Dmol.view(width=width, height=height)
        xyzview.addModel(xyz, "xyz")
        if debug:
            print(xyz)
        xyzview.setStyle({style: {}})
        # draw the box
        a0 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        from_ = [a0, a1 + a2, a1 + a3, a2 + a3]
        to_ = [
            [a1, a2, a3],
            [a1, a2, a1 + a2 + a3],
            [a1, a3, a1 + a2 + a3],
            [a2, a3, a1 + a2 + a3],
        ]
        for frm, li_to in zip(from_, to_):
            x0, y0, z0 = frm
            for to in li_to:
                x1, y1, z1 = to
                xyzview.addLine(
                    {
                        "color": "blue",
                        "start": {"x": x0, "y": y0, "z": z0},
                        "end": {"x": x1, "y": y1, "z": z1},
                    }
                )
        # show
        xyzview.zoomTo()
        xyzview.show()
