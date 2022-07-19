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

        sigmac_eigen_n_a_ref = np.array([[[0.11073960667658093,
            -2.474271605937808e-07, -1.3598224907549262e-07,
            -2.675923841305552e-07], [-2.474271605937808e-07,
            -0.10266828904501563, -3.9608333020684353e-07,
            2.644615657558003e-07], [-1.3598224907549262e-07,
            -3.9608333020684353e-07, -0.10266339860524415,
            -9.600464665648752e-08], [-2.675923841305552e-07,
            2.644615657558003e-07, -9.600464665648752e-08,
            -0.10266664292205206]]])
        sigmac_eigen_n_e_ref = np.array([[[-0.045624705527325685, 1.3037924217925383e-06, 
            -8.944029418909064e-06, 
            5.965224936866103e-07], [1.3037924217925383e-06, -0.056167116013829774,
            6.038646422297296e-06, -2.13718900591679e-06], [-8.944029418909064e-06,
            6.038646422297296e-06, -0.05618273223570602, -3.572766283146338e-06], 
            [5.965224936866103e-07,-2.13718900591679e-06, -3.572766283146338e-06,
            -0.05615252635858076]]])
        sigmac_eigen_n_f_ref = np.array([[[0.06511490114925514, 1.056365261198749e-06, 
            -9.08001166798455e-06, 3.289301095560312e-07], [1.056365261198749e-06, 
            -0.15883540505884536, 5.642563092090256e-06, -1.872727440160955e-06], 
            [-9.08001166798455e-06, 5.642563092090256e-06, -0.15884613084095023, 
            -3.6687709298028223e-06], [3.289301095560312e-07, -1.872727440160955e-06, 
            -3.6687709298028223e-06, -0.15881916928063278]]])

        np.testing.assert_almost_equal(self.qdetresult.sigmac_eigen_n_a,
                sigmac_eigen_n_a_ref)
        np.testing.assert_almost_equal(self.qdetresult.sigmac_eigen_n_e,
                sigmac_eigen_n_e_ref)
        np.testing.assert_almost_equal(self.qdetresult.sigmac_eigen_n_f,
                sigmac_eigen_n_f_ref)
    
