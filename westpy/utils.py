from __future__ import print_function

""" Set of utilities."""

def extractFileNamefromUrl(url):
   """Extracts a file name from url. 

   :param url: url
   :type url: string
   :returns: file name 
   :rtype: string

   :Example:

   >>> from westpy import * 
   >>> extractFileNamefromUrl("http://www.west-code.org/database/gw100/xyz/CH4.xyz")
   """
   #
   fname = None 
   my_url = url[:-1] if url.endswith('/') else url
   if my_url.find('/'):
      fname = my_url.rsplit('/', 1)[1]
   return fname 
    

def download(url,fname=None):
   """Downloads a file from url. 

   :param url: url
   :type url: string
   :param fname: file name, optional 
   :type fname: string

   :Example:

   >>> from westpy import * 
   >>> download("http://www.west-code.org/database/gw100/xyz/CH4.xyz")

   .. note:: The file will be downloaded in the current directory. 
   """
   #
   if fname is None :
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


def bool2str( logical ):
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
   if( logical ) : 
      return ".TRUE."
   else : 
      return ".FALSE."

def writeJsonFile(fname,data):
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
   with open(fname, 'w') as file:
      json.dump(data, file, indent=2)
      #
      print("")
      print("File written : ", fname )  

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
   with open(fname, 'r') as file:
      data = json.load(file)
      #
      print("")
      print("File read : ", fname )
   return data  

def convertYaml2Json(fyml,fjson):
   """Converts the file from YAML to JSON. 

   :param fyml: Name of YAML file 
   :type fyml: string
   :param fjson: Name of JSON file 
   :type fjson: string

   :Example:

   >>> from westpy import * 
   >>> convertYaml2Json("file.yml","file.json") 

   .. note:: The file fjon will be created, fyml will not be overwritten. 
   """
   #
   import yaml, json 
   from westpy import writeJsonFile
   #
   data = yaml.load(open(fyml))
   writeJsonFile(fjson,data)

def listLinesWithKeyfromOnlineText(url,key):
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
   import re
   data = urlopen(url) # parse the data
   greplist = []
   for line in data :
      if( key in str(line) ) : 
         greplist.append(line)
   return greplist

#
# list values from XML file located at url, with key  
#
def listValuesWithKeyFromOnlineXML(url,key):
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

   .. note:: Can be used to grep values from a XML file.
   """
   #
   from urllib.request import urlopen
   import xml.etree.ElementTree as ET
   tree = ET.parse(urlopen(url)) # parse the data
   root = tree.getroot()
   xml_values = [str(xml_val.text).strip() for xml_val in root.iter(key)] #get values
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
   return 1./(np.sqrt(2.*np.pi)*sig)*np.exp(-np.power((x - mu)/sig, 2.)/2)
