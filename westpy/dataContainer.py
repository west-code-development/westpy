from __future__ import print_function

class DataContainer() : 
   #
   """Class for representing an in-memory data container.
   
   :Example:

   >>> from westpy import * 
   >>> dc = DataContainer()
 
   """
   # 
   def __init__(self) :
      from signac import Collection
      self.info = {}
      self.coll = Collection()
   #
   def upsertPoint(self,identifier,document,incremental_update=True) : 
      """Update or inserts an entry to the data container. 

      If identifier exists, update the document associated to the identifier, otherwise insert the document with the identifier. 
   
      :param identifier: identifier
      :type key: * (hashable object) 
      :param document: document  
      :type document: * (hashable object)
      :param incremental_update: if the document exists, only update it, do not remove its other keys. 
      :type incremental_update: boolean
   
      :Example:

      >>> from westpy import *
      >>> dc = DataContainer()
      >>> dc.upsertPoint({"a":1, "b":2},{"energy":-4.5}) 
      """
      #
      # check if identifier exists, if yes update its document, otherwise plain insert
      #
      point = self.coll.find_one({"i" : identifier})
      if point is None : 
         self.coll.insert_one({"i" : identifier, "d" : document})
      else : 
         if( incremental_update ) : 
            for key in document.keys() :
               point["d"][key] = document[key]
            self.coll.replace_one({"i" : identifier},point)
         else : 
            self.coll.replace_one({"i" : identifier},{ "i" : identifier, "d" : document })
   #
   def showPoints(self) : 
      """Shows all points of the data container.  
   
      :Example:

      >>> from westpy import *
      >>> dc = DataContainer()
      >>> dc.upsertPoint({"a":1, "b":2},{"energy":-4.5}) 
      >>> dc.showPoints() 
      """
      for point in self.coll : 
         print(point)
   #   
   def removePoint(self,identifier) : 
      """Removes point with given identifier from the data container.  
   
      :param identifier: identifier
      :type key: * (hashable object) 
   
      :Example:

      >>> from westpy import *
      >>> dc = DataContainer()
      >>> dc.upsertPoint({"a":1, "b":2},{"energy":-4.5}) 
      >>> dc.removePoint({"a":1, "b":2}) 
      """
      #
      # check if identifier exists, if yes remove the point
      #
      self.coll.delete_one({"i" : identifier})
   #
   def upsertKey(self,key,description):
      """Updates or inserts a new key and its description.  
   
      :param key: key
      :type key: string
      :param description: description
      :type description: * (hashable object)

      :Example:

      >>> from westpy import *
      >>> dc = DataContainer()
      >>> dc.upsertKey("a","the first letter") 
      """
      #
      # if key exists update its description, otherwise insert
      #
      self.info[key] = description
   #
   def removeKey(self,key):
      """Removes the description of a key  
   
      :param key: key
      :type key: string
   
      :Example:

      >>> from westpy import *
      >>> dc = DataContainer()
      >>> dc.upsertKey("a","the first letter") 
      >>> dc.removeKey("a") 
      """
      #
      # if key exists, remove it
      #
      if key in self.info.keys() : 
         self.info.pop(key)
   #
   def checkKeys(self,printSummary=True):
      """Checks that all keys are described.

      :param printSummary: if True prints summary
      :type printSummary: boolean
      :returns: True if all keys are described, False otherwise. 
      :rtype: boolean
   
      :param key: key
      :type key: string
   
      :Example:

      >>> from westpy import *
      >>> dc = DataContainer()
      >>> dc.upsertPoint({"a":1, "b":2},{"energy":-4.5}) 
      >>> dc.upsertKey("a","the first letter") 
      >>> dc.upsertKey("b","another letter") 
      >>> dc.upsertKey("energy","a quantity")
      >>> dc.checkKeys() 
      """
      #
      undesc_keys = []
      utilized_keys = []
      unutilized_keys = []
      #
      for point in self.coll : 
          for key in point["i"].keys() : 
             if key not in utilized_keys : 
                utilized_keys.append(key)
             if key not in self.info.keys() : 
                if key not in undesc_keys : 
                   undesc_keys.append(key)
          for key in point["d"].keys() : 
             if key not in utilized_keys : 
                utilized_keys.append(key)
             if key not in self.info.keys() : 
                if key not in undesc_keys : 
                   undesc_keys.append(key)
      #
      for key in self.info.keys() : 
         if key not in utilized_keys : 
            unutilized_keys.append(key)
      # print summary
      if( printSummary ) : 
         if( len(self.info.keys()) == 0 ) :
            print( "Described keys : None") 
         for key in self.info.keys() : 
            print( "Described key ... ", key, " : ", self.info[key] )
         if( len( undesc_keys ) > 0 ) : 
            print( "Undescribed keys ... ", undesc_keys )
         if( len( unutilized_keys ) > 0 ) : 
            print( "Unutilized keys ... ", unutilized_keys )
      #
      return len(undesc_keys) == 0
