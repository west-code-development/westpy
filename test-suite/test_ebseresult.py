import unittest
import json
from westpy.qdet import eBSEResult
from pathlib import Path
import numpy as np

class eBSEResultTestCase(unittest.TestCase):
    def setUp(self):
        # generate and store eBSEResult object
        self.path = str(Path('./data').resolve() / 'wfreq_spinpol.json')
        self.ebseresult = eBSEResult(filename=self.path, spin_flip_=True)
        
        # read and store reference data from JSON
        ref_path = str(Path('./data').resolve() / 'ebse_reference.json')
        with open(ref_path, 'r') as f:
            ref_data = json.load(f)
        self.ebse_ref = ref_data
        # read and store reference for solved eBSE from JSON
        ref_path = str(Path('./data').resolve() / 'ebse_solution_reference.json')
        with open(ref_path, 'r') as f:
            ref_data = json.load(f)
        self.solution_ref = ref_data


    def test_class_variables(self):
        """
        Tests that eBSE class variables are correct.
        """
        self.assertEqual(self.ebseresult.filename, str(Path('./data').resolve() / 'wfreq_spinpol.json'))
        # test all scalar parameters
        self.assertEqual(self.ebseresult.spin_flip, self.ebse_ref['spin_flip'])
        self.assertEqual(self.ebseresult.n_orbitals, self.ebse_ref['n_orbitals'])
        self.assertEqual(self.ebseresult.nelec, self.ebse_ref['nelec'])
        self.assertEqual(self.ebseresult.n_tr, self.ebse_ref['n_tr'])

        #test array parameters
        self.assertListEqual(self.ebseresult.basis.tolist(),self.ebse_ref['basis'])
        self.assertListEqual(self.ebseresult.qp_energies.tolist(),self.ebse_ref['qp_energies'])
        self.assertListEqual(self.ebseresult.occ.tolist(),self.ebse_ref['occ'])
        self.assertListEqual(self.ebseresult.v.tolist(),self.ebse_ref['v'])
        self.assertListEqual(self.ebseresult.w.tolist(),self.ebse_ref['w'])
        self.assertListEqual(self.ebseresult.smap.tolist(),self.ebse_ref['smap'])
        self.assertListEqual(self.ebseresult.cmap.tolist(),self.ebse_ref['cmap'])
        self.assertListEqual(self.ebseresult.jwstring.tolist(),self.ebse_ref['jwstring'])
    
    def test_solution(self):
        """
        Tests the result of the eBSE diagonalization.
        """
        # solve eBSE Hamiltonian
        solution = self.ebseresult.solve(verbose=False)
        
        np.testing.assert_almost_equal(solution['hamiltonian'],
                np.array(self.solution_ref['hamiltonian']))
        np.testing.assert_almost_equal(solution['evs'],
                np.array(self.solution_ref['evs']))
        np.testing.assert_almost_equal(solution['evs_au'],
                np.array(self.solution_ref['evs_au']))
        np.testing.assert_almost_equal(solution['evcs'],
                np.array(self.solution_ref['evcs']))
        np.testing.assert_almost_equal(solution['rdm1s'],
                np.array(self.solution_ref['rdm1s']))
        np.testing.assert_almost_equal(solution['mults'],
                np.array(self.solution_ref['mults']))

