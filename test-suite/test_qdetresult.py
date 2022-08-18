import unittest
import json
from westpy.qdet import QDETResult
from pathlib import Path
import numpy as np

class QDETResultTestCase(unittest.TestCase):
    def setUp(self):
        # generate and store QDETResult object
        self.path = str(Path('./data').resolve() / 'wfreq.json')
        self.qdetresult = QDETResult(filename=self.path)
        
        # read and store reference data from JSON
        ref_path = str(Path('./data').resolve() / 'ref.json')
        with open(ref_path, 'r') as f:
            ref_data = json.load(f)
        self.ref = ref_data

    def test_class_variables(self):
        """
        Tests QDET parameters are correct.
        """
        self.assertEqual(self.qdetresult.filename, str(Path('./data').resolve() / 'wfreq.json'))
        self.assertEqual(self.qdetresult.point_group, None)
        self.assertEqual(self.qdetresult.nspin, 1)
        self.assertListEqual(self.qdetresult.basis.tolist(), [87, 122, 123, 126, 127, 128])
        self.assertEqual(self.qdetresult.npair, 21)
        self.assertEqual(self.qdetresult.occupation.tolist(), [[2.0, 2.0, 2.0, 2.0, 1.0, 1.0]])
    
    def test_h1e(self):
        """
        Test QDET one-body terms.
        """
        np.testing.assert_almost_equal(self.qdetresult.h1e, np.array(self.ref['h1e']))

    def test_eri(self):
        """
        Test QDET two-body terms.
        """

        np.testing.assert_almost_equal(self.qdetresult.eri, np.array(self.ref['eri']))

    def test_solution(self):
        """
        Test QDET eigenvalues.
        """
        solution = self.qdetresult.solve()

        np.testing.assert_almost_equal(solution['evs'], np.array([0.0, 0.43604111, 0.436138, 1.25034936, 
            1.94063497, 1.94070084, 2.93681169, 2.93688193, 4.66194716, 5.07277312]))


