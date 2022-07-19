import unittest
from westpy.qdet import QDETResult, Heff
from pathlib import Path
import numpy as np
import json

class HeffTestCase(unittest.TestCase):
    def setUp(self):
        # read parameter data from JSON file
        with open('./data/HeffParameter.json', 'r') as f:
            self.parameters = json.load(f)
        self.heff = Heff(h1e=np.array(self.parameters['h1e']),
                     eri=np.array(self.parameters['eri']))

    def test_class_variables(self):
        """
        Test class variables.
        """
        self.assertEqual(self.heff.nspin, 1)
        self.assertEqual(self.heff.norb, 7)
    
    def test_FCI(self):
        """
        Test FCI solution of the effective Hamiltonian.
        """
        results_ = self.heff.FCI(nelec=(1,1))
        # test excitation energies
        np.testing.assert_almost_equal(results_['evs'], self.parameters['evs'])

        evcs_ = np.array(self.parameters['evcs'])
        results_['evcs'] = np.asarray(results_['evcs'])

        np.testing.assert_almost_equal(results_['evcs'], evcs_)

        np.testing.assert_almost_equal(results_['mults'],
                np.array(self.parameters['mults']))
       
        np.testing.assert_almost_equal(results_['rdm1s'],
                np.array(self.parameters['rdm1s']))

        
