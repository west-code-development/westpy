""" Set of utilities."""


def extractFileNamefromUrl(url):
    """Extracts a file name from url.

    :param url: url
    :type url: string
    :returns: file name
    :rtype: string

    :Example:

    >>> from westpy import *
    >>> extractFileNamefromUrl("https://west-code.org/database/gw100/xyz/CH4.xyz")
    """
    #
    fname = None
    my_url = url[:-1] if url.endswith("/") else url
    if my_url.find("/"):
        fname = my_url.rsplit("/", 1)[1]
    return fname


def download(url, fname=None):
    """Downloads a file from url.

    :param url: url
    :type url: string
    :param fname: file name, optional
    :type fname: string

    :Example:

    >>> from westpy import *
    >>> download("https://west-code.org/database/gw100/xyz/CH4.xyz")

    .. note:: The file will be downloaded in the current directory.
    """
    #
    if fname is None:
        fname = extractFileNamefromUrl(url)
    #
    from requests import get

    # open in binary mode
    with open(fname, "wb") as file:
        # get request
        response = get(url)
        # write to file
        file.write(response.content)
        #
        print("Downloaded file: ", fname, ", from url: ", url)


def bool2str(logical):
    """Converts a boolean type into a string .TRUE. or .FALSE. .

    :param logical: logical
    :type logical: boolean
    :returns: .TRUE. or .FALSE.
    :rtype: string

    :Example:

    >>> from westpy import *
    >>> t = bool2str(True)
    >>> f = bool2str(False)
    >>> print(t,f)
    .TRUE. .FALSE.
    """
    #
    if logical:
        return ".TRUE."
    else:
        return ".FALSE."


def writeJsonFile(fname, data):
    """Writes data to file using the JSON format.

    :param fname: file name
    :type fname: string
    :param data: data
    :type data: dict/list

    :Example:

    >>> from westpy import *
    >>> data = {}
    >>> data["mass"] = 1.0
    >>> writeJsonFile("mass.json",data)

    .. note:: The file will be generated in the current directory.
    """
    #
    import json

    #
    with open(fname, "w") as file:
        json.dump(data, file, indent=2)
        #
        print("")
        print("File written : ", fname)


def readJsonFile(fname):
    """Reads data from file using the JSON format.

    :param fname: file name
    :type fname: string
    :returns: data
    :rtype: dict/list

    :Example:

    >>> from westpy import *
    >>> data = readJsonFile("mass.json")

    .. note:: The file will be read from the current directory.
    """
    #
    import json

    #
    with open(fname, "r") as file:
        data = json.load(file)
        #
        print("")
        print("File read : ", fname)
    return data


def convertYaml2Json(fyml, fjson):
    """Converts the file from YAML to JSON.

    :param fyml: Name of YAML file
    :type fyml: string
    :param fjson: Name of JSON file
    :type fjson: string

    :Example:

    >>> from westpy import *
    >>> convertYaml2Json("file.yml","file.json")

    .. note:: The file fjson will be created, fyml will not be overwritten.
    """
    #
    import yaml
    from westpy import writeJsonFile

    #
    data = yaml.load(open(fyml))
    writeJsonFile(fjson, data)


def listLinesWithKeyfromOnlineText(url, key):
    """List lines from text file located at url, with key.

    :param url: url
    :type url: string
    :param key: key word
    :type key: string
    :returns: list of lines
    :rtype: list

    :Example:

    >>> from westpy import *
    >>> url = "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.1.upf"
    >>> key = "z_valence"
    >>> l = listLinesWithKeyfromOnlineText(url,key)
    >>> print(l)
    ['       z_valence="    4.00"']

    .. note:: Can be used to grep values from a UPF file.
    """
    #
    from urllib.request import urlopen

    data = urlopen(url)  # parse the data
    greplist = []
    for line in data:
        if key in str(line):
            greplist.append(line)
    return greplist


def listValuesWithKeyFromOnlineXML(url, key):
    """List values from XML file located at url, with key.

    :param url: url
    :type url: string
    :param key: key word
    :type key: string
    :returns: list of values
    :rtype: list

    :Example:

    >>> from westpy import *
    >>> url = "http://www.quantum-simulation.org/potentials/sg15_oncv/xml/Si_ONCV_PBE-1.1.xml"
    >>> key = "valence_charge"
    >>> l = listLinesWithKeyfromOnlineXML(url,key)
    >>> print(l)
    ['4']

    .. note:: Can be used to grep values from an XML file.
    """
    #
    from urllib.request import urlopen
    import xml.etree.ElementTree as ET

    tree = ET.parse(urlopen(url))  # parse the data
    root = tree.getroot()
    xml_values = [str(xml_val.text).strip() for xml_val in root.iter(key)]  # get values
    return xml_values


def gaussian(x, mu, sig):
    """return normal distribution at point x.

    :math:`f(x;\\mu,\\sigma) = \\frac{1}{\\sigma\sqrt{2\\pi}}e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}`

    :param x: x
    :type x: float
    :param mu: :math:`\\mu`
    :type mu: float
    :param sigma: :math:`\\sigma`
    :type sigma: float
    :returns: :math:`f(x;\\mu,\\sigma)`
    :rtype: float

    :Example:

    >>> from westpy import *
    >>> gaussian(1.0,2.0,3.0)
    """
    import numpy as np

    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )


def _putline(*args):
    """
    Generate a line to be written to a cube file where
    the first field is an int and the remaining fields are floats.

    params:
        *args: first arg is formatted as int and remaining as floats

    returns: formatted string to be written to file with trailing newline
    """
    s = "{0:^ 8d}".format(args[0])
    s += "".join("{0:< 12.6f}".format(arg) for arg in args[1:])
    return s + "\n"


def _getline(cube):
    """
    Read a line from cube file where first field is an int
    and the remaining fields are floats.

    params:
        cube: file object of the cube file

    returns: (int, list<float>)
    """
    l = cube.readline().strip().split()
    return int(l[0]), map(float, l[1:])


def read_cube(fname):
    """
    Read cube file into numpy array

    :param fname: filename of cube file
    :type fname: string
    :returns: (data, metadata)
    :rtype: (np.array, dict)
    """
    import numpy as np

    meta = {}
    with open(fname, "r") as cube:
        cube.readline()
        cube.readline()  # ignore comments
        natm, meta["org"] = _getline(cube)
        nx, meta["xvec"] = _getline(cube)
        ny, meta["yvec"] = _getline(cube)
        nz, meta["zvec"] = _getline(cube)
        meta["atoms"] = [_getline(cube) for i in range(natm)]
        data = np.zeros((nx * ny * nz))
        idx = 0
        for line in cube:
            for val in line.strip().split():
                data[idx] = float(val)
                idx += 1
    data = np.reshape(data, (nx, ny, nz))
    return data, meta


def read_imcube(rfname, ifname=""):
    """
    Convenience function to read in two cube files at once,
    where one contains the real part and the other contains the
    imag part. If only one filename given, other filename is inferred.

    :param rfname: filename of cube file of real part
    :type rfname: string
    :param ifname: optional, filename of cube file of imag part
    :type fname: string
    :returns: (data, metadata), where data is (real part + j*imag part)
    :rtype: (np.array, dict)
    """
    import numpy as np

    ifname = ifname or rfname.replace("real", "imag")
    _debug("reading from files", rfname, "and", ifname)
    re, im = read_cube(rfname), read_cube(ifname)
    fin = np.zeros(re[0].shape, dtype="complex128")
    if re[1] != im[1]:
        _debug("warning: meta data mismatch, real part metadata retained")
    fin += re[0]
    fin += 1j * im[0]
    return fin, re[1]


def write_cube(data, meta, fname):
    """
    Write volumetric data to cube file along

    :param data: volumetric data consisting real values
    :type data: list of float
    :param meta: dict containing metadata with following keys:

        - atoms: list of atoms in the form (mass, [position])
        - org: origin
        - xvec,yvec,zvec: lattice vector basis
    :type meta: dict
    :param fname: filename of cubefile (existing files overwritten)
    :type fname: string
    """
    with open(fname, "w") as cube:
        # first two lines are comments
        cube.write(" Cubefile created by cubetools.py\n  source: none\n")
        natm = len(meta["atoms"])
        nx, ny, nz = data.shape
        cube.write(_putline(natm, *meta["org"]))  # 3rd line #atoms and origin
        cube.write(_putline(nx, *meta["xvec"]))
        cube.write(_putline(ny, *meta["yvec"]))
        cube.write(_putline(nz, *meta["zvec"]))
        for atom_mass, atom_pos in meta["atoms"]:
            cube.write(_putline(atom_mass, *atom_pos))  # skip the newline
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    if (i or j or k) and k % 6 == 0:
                        cube.write("\n")
                    cube.write(" {0: .5E}".format(data[i, j, k]))


def write_imcube(data, meta, rfname, ifname=""):
    """
    Convenience function to write two cube files from complex valued
    volumetric data, one for the real part and one for the imaginary part.
    Data about atoms, origin and lattice vectors are kept same for both.
    If only one filename given, other filename is inferred.

    :param data: volumetric data consisting complex values
    :type data: list of complex
    :param meta: dict containing metadata with following keys:

        - atoms: list of atoms in the form (mass, [position])
        - org: origin
        - xvec,yvec,zvec: lattice vector basis
    :type meta: dict
    :param rfname: filename of cubefile containing real part
    :type rfname: string
    :param ifname: optional, filename of cubefile containing imag part
    :type ifname: string
    """
    ifname = ifname or rfname.replace("real", "imag")
    _debug("writing data to files", rfname, "and", ifname)
    write_cube(data.real, meta, rfname)
    write_cube(data.imag, meta, ifname)


def wfreq2df(
    fname="wfreq.json",
    dfKeys=[
        "eks",
        "eqpLin",
        "eqpSec",
        "sigmax",
        "sigmac_eks",
        "sigmac_eqpLin",
        "sigmac_eqpSec",
        "vxcl",
        "vxcnl",
        "hf",
    ],
):
    """
    Loads the wfreq JSON output into a pandas dataframe.

    :param fname: filename of JSON output file
    :type fname: string
    :param dfKeys: energy keys to be added to dataframe
    :type dfKeys: list of string
    :returns: (dataframe, data)
    :rtype: (pd.DataFrame, dict)
    """
    #
    import json

    #
    with open(fname) as file:
        data = json.load(file)
    #
    import numpy as np
    import pandas as pd

    #
    # build dataframe
    #
    cols = ["k", "s", "n"] + dfKeys
    df = pd.DataFrame(columns=cols)
    #
    # insert data into dataframe
    #
    j = 0
    for s in range(1, data["system"]["electron"]["nspin"] + 1):
        for k in data["system"]["bzsamp"]["k"]:
            kindex = f"K{k['id']+(s-1)*len(data['system']['bzsamp']['k']):06d}"
            d = data["output"]["Q"][kindex]
            for i, n in enumerate(data["input"]["wfreq_control"]["qp_bands"][s - 1]):
                row = [k["id"], s, n]
                for key in dfKeys:
                    if "re" in d[key]:
                        row.append(d[key]["re"][i])
                    else:
                        row.append(d[key][i])
                df.loc[j] = row
                j += 1
    #
    # cast columns k, s, n to int
    #
    for col in ["k", "s", "n"]:
        df[col] = df[col].apply(np.int64)

    return df, data
