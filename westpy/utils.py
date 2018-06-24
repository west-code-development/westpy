from __future__ import print_function

""" Set of utilities."""

def download(fname, url):
   """Downloads a file from url. 

   :param fname: file name
   :type fname: string
   :param url: url
   :type url: string

   :Example:

   >>> from westpy import * 
   >>> download("CH4.xyz","http://www.west-code.org/database/gw100/xyz/CH4.xyz")

   .. note:: The file will be downloaded in the current directory. 
   """
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

def writeJsonFile(data,fname):
   """Writes data to file using the JSON format. 

   :param data: data
   :type data: dict
   :param fname: file name
   :type fname: string

   :Example:

   >>> from westpy import * 
   >>> data = {}
   >>> data["mass"] = 1.0
   >>> writeJsonFile(data,"mass.json") 

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
