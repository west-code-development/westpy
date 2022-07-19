import unittest
from westpy.qdet import QDETResult, Heff
from pathlib import Path
import numpy as np

class HeffTestCase(unittest.TestCase):
    def setUp(self):
        path = str(Path('./data').resolve())
        qdetresult = QDETResult(path)

