import unittest
from westpy.qdet import QDETResult
from pathlib import Path
import numpy as np

class QDETResultTestCase(unittest.TestCase):
    def setUp(self):
        self.path = str(Path('./data').resolve())
        self.qdetresult = QDETResult(path=self.path)


    def test_class_variables(self):
        """
        Tests QDET parameters are correct.
        """
        self.assertEqual(self.qdetresult.path, str(Path('./data').resolve()))
        self.assertEqual(self.qdetresult.eps_infty, None)
        self.assertEqual(self.qdetresult.point_group, None)
        self.assertEqual(self.qdetresult.nspin, 1)
        self.assertEqual(self.qdetresult.npdep, 122)
        self.assertEqual(self.qdetresult.omega, 1035.521592416925)
        self.assertEqual(self.qdetresult.nproj, 4)
        self.assertListEqual(self.qdetresult.ks_projectors.tolist(), [61, 63, 64, 65])
        self.assertListEqual(self.qdetresult.basis.tolist(), [0, 1, 2, 3])
        self.assertEqual(self.qdetresult.npair, 10)
        self.assertEqual(self.qdetresult.div, 0.4905396)
        self.assertEqual(self.qdetresult.nel, 2)

    def test_quasiparticle_properties(self):
        """
        Test quasiparticle energies, self-energies and occupations.
        """
        qp_energies_ref = np.array([0.2961498, 0.62429574, 0.62428618, 0.62430392])
        occ_ref = np.array([[2.0, 0.0, 0.0, 0.0]])
        sigmax_n_a_ref = np.array([[[-0.3328455031760298, 8.559391576675596e-08, 
            -1.1561204530437935e-07, 7.925646267926967e-08], 
            [8.559391576675596e-08, -0.009970049134502543, 
            -1.0833799347844016e-07, -2.9786915537724154e-07], 
            [-1.1561204530437935e-07, -1.0833799347844016e-07, 
            -0.009969704037751126, -1.7378722689371205e-07], 
            [7.925646267926967e-08, -2.9786915537724154e-07, 
            -1.7378722689371205e-07, -0.00996999206278382]]])
        sigmax_n_e_ref =np.array([[[-0.13048604804724517, -2.898233602459193e-07, 
            2.830887511417004e-07, -3.982076464479744e-07], 
            [-2.898233602459193e-07, -0.19997720008997139, 1.3758348700945722e-07, 
            -5.614265981008625e-07], [2.830887511417004e-07, 1.3758348700945722e-07, 
            -0.19997697375534892, -1.1115221148479213e-06], [-3.982076464479744e-07, 
            -5.614265981008625e-07, -1.1115221148479213e-06,
            -0.19997752424240584]]])
        sigmax_n_f_ref = np.array([[[-0.4633315512232749, -2.0422944447921754e-07, 
            1.6747670583856789e-07, -3.189511837663195e-07], 
            [-2.0422944447921754e-07, -0.20994724922447394, 2.924549353101706e-08, 
            -8.592957534787546e-07], [1.6747670583856789e-07, 2.924549353101706e-08, 
            -0.20994667779310006, -1.2853093417416334e-06], [-3.189511837663195e-07, 
            -8.592957534787546e-07, -1.2853093417416334e-06,
            -0.20994751630518962]]])
            
        np.testing.assert_almost_equal(np.diagonal(self.qdetresult.qp_energy_n[0]),
                qp_energies_ref)
        np.testing.assert_almost_equal(self.qdetresult.occ, occ_ref)
        
        np.testing.assert_almost_equal(self.qdetresult.sigmax_n_a, sigmax_n_a_ref)
        np.testing.assert_almost_equal(self.qdetresult.sigmax_n_e, sigmax_n_e_ref)
        np.testing.assert_almost_equal(self.qdetresult.sigmax_n_f, sigmax_n_f_ref)
