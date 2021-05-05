from __future__ import absolute_import, division, print_function

from westpy.embedding import *
from westpy.units import * 
from westpy.utils import * 
from westpy.atom import * 
from westpy.geometry import * 
from westpy.groundState import * 
from westpy.dataContainer import * 
from westpy.electronicStructure import * 
from westpy.session import *

__version__ = '3.1.1'

def header() :
   """Prints welcome header."""
   import datetime
   print(" ")
   print(" _    _ _____ _____ _____            ")
   print("| |  | |  ___/  ___|_   _|           ")
   print("| |  | | |__ \ `--.  | |_ __  _   _  ")
   print("| |/\| |  __| `--. \ | | '_ \| | | | ")
   print("\  /\  / |___/\__/ / | | |_) | |_| | ")
   print(" \/  \/\____/\____/  \_/ .__/ \__, | ")
   print("                       | |     __/ | ")
   print("                       |_|    |___/  ")
   print(" ")
   print("WEST version     : ",__version__)
   print("Today            : ", datetime.datetime.today())

try:
   from mpi4py import MPI
   comm = MPI.COMM_WORLD
   rank = comm.rank
   if rank == 0:
      header()
except:
   header()
