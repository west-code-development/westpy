import unittest
import json
from westpy.qdet import (
    PointGroup,
    PointGroupOperation,
    PointGroupRotation,
    PointGroupReflection,
)
from westpy.qdet import VData
from pathlib import Path
import numpy as np


class SymmetryTestCase(unittest.TestCase):
    def setUp(self):
        # read Kohn-Sham wavefunctions from cube files
        filenames = [
            str(Path("./data").resolve() / "wfcK000001B000126.cube"),
            str(Path("./data").resolve() / "wfcK000001B000127.cube"),
            str(Path("./data").resolve() / "wfcK000001B000128.cube"),
        ]

        self.orbitals = [VData(entry, normalize="sqrt") for entry in filenames]

        # generate Point Group
        sq3 = np.sqrt(3)

        origin = 27 * np.array([0.48731000, 0.48731000, 0.48731000])

        self.point_group = PointGroup(
            name="C3v",
            operations={
                "E": PointGroupOperation(T=np.eye(4)),
                "C3_1": PointGroupRotation(
                    rotvec=2 * np.pi / 3 * np.array([1 / sq3, 1 / sq3, 1 / sq3]),
                    origin=origin,
                ),
                "C3_2": PointGroupRotation(
                    rotvec=4 * np.pi / 3 * np.array([1 / sq3, 1 / sq3, 1 / sq3]),
                    origin=origin,
                ),
                "Cv_1": PointGroupReflection(normal=(1, -1, 0), origin=origin),
                "Cv_2": PointGroupReflection(normal=(0, -1, 1), origin=origin),
                "Cv_3": PointGroupReflection(normal=(-1, 0, 1), origin=origin),
            },
            ctable={
                "A1": [1, 1, 1, 1, 1, 1],
                "A2": [1, 1, 1, -1, -1, -1],
                "E": [2, -1, -1, 0, 0, 0],
            },
        )

        # read reference data to dictionary
        with open(str(Path("./data/symm_ref.json").resolve()), "r") as f:
            self.ref_data = json.load(f)

    def test_point_group_rep(self):
        """
        Test whether point group representation for set of orbitals is correct.
        """

        point_group_rep = self.point_group.compute_rep_on_orbitals(
            self.orbitals, orthogonalize=True
        )[0]

        for key in point_group_rep.rep_matrices.keys():
            np.testing.assert_almost_equal(
                point_group_rep.rep_matrices[key],
                np.array(self.ref_data["PointGroupRep"][key]),
            )

    def test_orbital_symms(self):
        """
        Test symmetries for each orbital.
        """

        orbital_symms = self.point_group.compute_rep_on_orbitals(
            self.orbitals, orthogonalize=True
        )[1]

        for i in range(len(orbital_symms)):
            self.assertEqual(orbital_symms[i], self.ref_data["orbital_symms"][i])
