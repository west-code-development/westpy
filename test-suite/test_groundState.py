import unittest
from westpy import Geometry, GroundState

class GroundStateTestCase(unittest.TestCase):
    def setUp(self):
        geom = Geometry()
        geom.setCell((1, 0, 0), (0, 1, 0), (0, 0, 1))
        geom.addAtom("Si", (0, 0, 0))
        geom.addSpecies(
            "Si",
            "http://www.quantum-simulation.org/potentials/sg15_oncv/upf/Si_ONCV_PBE-1.2.upf",
        )
        gs = GroundState(geom, "PBE", 30.0)
        gs.generateInputPW("pw.in")

    def test_pw_input(self):
        # read reference file
        with open("./data/pw_ref.in", "r") as f1:
            c1 = f1.read()

        # read test file
        with open("./pw.in", "r") as f2:
            c2 = f2.read()

        assert c1 == c2
