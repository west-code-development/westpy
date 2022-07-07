import unittest
from westpy.qdet import QDETResults
from pathlib import Path
import numpy as np

class QDETResultTestCase(unittest.TestCase):
    def setUp(self):
        self.path = str(Path('./data').resolve)
        self.qdetresults = QDETResults(path=path)


    def test_class_variables(self):
        """
        Tests QDET parameters are correct.
        """
        self.assertEqual(self.path, str(Path('./data')))
        self.assertEqual(self.eps_infty, None)
        self.assertEqual(point_group, None)
        self.assertEqual(self.nspin, 1)
        self.assertEqual(self.npdep, 506)
        self.assertEqual(self.omega, 4142.0863696677025)
        self.assertEqual(self.nproj, 6)
        self.assertEqual(self.ks_projectors, [252, 253, 254, 255, 256, 257])
        self.assertEqual(self.basis, [0, 1, 2, 3, 4, 5])
        self.assertEqual(self.npair, 21)
        self.assertEqual(self.div, 0.3042077)
        self.assertEqual(self.nel, 4)

    def test_quasiparticle_properties(self):
        """
        Test quasiparticle energies, self-energies and occupations.
        """
        qp_energies_ref = np.array([0.24890098, 0.36256301, 0.53116984,
            0.59000089, 0.58999846, 0.59000193])
        occ_ref = np.array([[2.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
        sigmax_n_a_ref = np.array([[[-1.54287804e-01, -4.58927888e-08,  9.05621209e-09,
          6.35364265e-04,  1.87518938e-04,  7.46113173e-06],
        [-4.58927888e-08, -2.61935666e-01, -4.27645531e-02,
          3.09108694e-07,  2.24444799e-07, -3.17017733e-06],
        [ 9.05621209e-09, -4.27645531e-02, -1.70653973e-02,
          3.87574396e-08,  1.33883450e-08, -4.29209105e-07],
        [ 6.35364265e-04,  3.09108694e-07,  3.87574396e-08,
         -3.95057902e-02,  1.40544824e-04, -6.90936168e-05],
        [ 1.87518938e-04,  2.24444799e-07,  1.33883450e-08,
          1.40544824e-04, -3.97155416e-02, -1.83632888e-04],
        [ 7.46113173e-06, -3.17017733e-06, -4.29209105e-07,
         -6.90936168e-05, -1.83632888e-04, -3.85625254e-02]]])
        sigmax_n_e_ref = np.array([[[-7.81071979e-01,  4.42349889e-07,  1.64820700e-07,
          2.25599658e-02,  5.53495461e-03,  1.27526535e-03],
        [ 4.42349889e-07, -2.47742653e-01,  2.76442867e-02,
         -3.15977779e-09,  4.04383932e-08,  4.96289727e-08],
        [ 1.64820700e-07,  2.76442867e-02, -3.02388408e-01,
         -4.49144083e-08, -3.43249719e-08,  3.17198596e-07],
        [ 2.25599658e-02, -3.15977779e-09, -4.49144083e-08,
         -2.63868745e-01, -1.40564410e-04,  6.92451897e-05],
        [ 5.53495461e-03,  4.04383932e-08, -3.43249719e-08,
         -1.40564410e-04, -2.63658926e-01,  1.83579588e-04],
        [ 1.27526535e-03,  4.96289727e-08,  3.17198596e-07,
          6.92451897e-05,  1.83579588e-04, -2.64812867e-01]]])
        sigmax_n_f_ref = np.array(array([[[-9.35359783e-01,  3.96457100e-07,  1.73876912e-07,
          2.31953300e-02,  5.72247354e-03,  1.28272648e-03],
        [ 3.96457100e-07, -5.09678319e-01, -1.51202665e-02,
          3.05948917e-07,  2.64883192e-07, -3.12054836e-06],
        [ 1.73876912e-07, -1.51202665e-02, -3.19453805e-01,
         -6.15696863e-09, -2.09366269e-08, -1.12010509e-07],
        [ 2.31953300e-02,  3.05948917e-07, -6.15696863e-09,
         -3.03374535e-01, -1.95861592e-08,  1.51572807e-07],
        [ 5.72247354e-03,  2.64883192e-07, -2.09366269e-08,
         -1.95861592e-08, -3.03374467e-01, -5.33001524e-08],
        [ 1.28272648e-03, -3.12054836e-06, -1.12010509e-07,
          1.51572807e-07, -5.33001524e-08, -3.03375392e-01]]]))

        np.testing.assert_almost_equal(np.diagonal(self.qp_energies),
                qp_energies_ref)
        np.testing.assert_almost_equal(self.occ, occ_ref)
        
        np.testing.assert_almost_equal(self.sigmax_n_a, sigmax_n_a_ref)
        np.testing.assert_almost_equal(self.sigmax_n_e, sigmax_n_e_ref)
        np.testing.assert_almost_equal(self.sigmax_n_f, sigmax_n_f_ref)
    def test_dft_properties(self):
        """
        Test DFT eigenvalues.
        """
