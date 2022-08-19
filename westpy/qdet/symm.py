from __future__ import annotations
from typing import Sequence, Union, Tuple, List, Dict
import numpy as np
from scipy.linalg import block_diag
from scipy.ndimage import affine_transform

try:
    from scipy.spatial.transform import Rotation
except:
    pass
from scipy.linalg import fractional_matrix_power

from .vdata import VData


class PointGroupOperation:
    def __init__(
        self, T: np.ndarray, origin: Union[Sequence[float], np.ndarray] = None
    ):
        """An operation in the point group.

        Args:
            T: 4x4 affine transformation matrix for point group operation
            origin: origin of operation
        """
        assert T.shape == (4, 4)
        self.T = T
        if origin is not None:
            assert len(origin) == 3
            self.set_origin(origin)

    def set_origin(self, origin):
        """Set origin of the operation.

        translation matrix TR
        (0, 0, 0) -> (x0, y0, z0)

        Args:
            origin: coordinate of origin.
        """

        x0, y0, z0 = origin
        TR = np.array(
            [
                [1, 0, 0, x0],
                [0, 1, 0, y0],
                [0, 0, 1, z0],
                [0, 0, 0, 1],
            ]
        )
        self.T = TR @ self.T @ np.linalg.inv(TR)

    @property
    def inv(self):
        """Inverse operator"""
        return PointGroupOperation(T=np.linalg.inv(self.T))

    def __matmul__(self, other: PointGroupOperation) -> PointGroupOperation:
        """Composite operator"""
        assert isinstance(other, PointGroupOperation)
        return PointGroupOperation(T=self.T @ other.T)

    def __call__(self, f: np.ndarray) -> np.ndarray:
        """Apply the operator to transform a function.

        Args:
            f: a real space function (volumetric data) to be transformed

        Returns:
            transformed function
        """
        return affine_transform(f, matrix=self.T)


class PointGroupReflection(PointGroupOperation):
    def __init__(
        self,
        normal: Union[Sequence[float], np.ndarray],
        origin: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
    ):
        """Reflection operation.

        reflection matrix RE
        plane: ax + by + cz = 0

        Args:
            normal: normal vector
            origin: origin
        """
        a, b, c = np.array(normal) / np.linalg.norm(normal)
        RE = np.array(
            [
                [1 - 2 * a * a, -2 * a * b, -2 * a * c, 0],
                [-2 * a * b, 1 - 2 * b * b, -2 * b * c, 0],
                [-2 * a * c, -2 * b * c, 1 - 2 * c * c, 0],
                [0, 0, 0, 1],
            ]
        )
        super(PointGroupReflection, self).__init__(T=RE, origin=origin)


class PointGroupRotation(PointGroupOperation):
    def __init__(
        self,
        rotvec: Union[Sequence[float], np.ndarray],
        origin: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
    ):
        """Rotation operation.

        reflection matrix RO

        Args:
            normal: rotvec: (a, b, c), |(a, b, c)| is interpreted as degree in radian, direction is interpreted as axis
            origin: origin
        """
        rotation = Rotation.from_rotvec(rotvec)
        RO = block_diag(rotation.as_matrix().T, 1)
        super(PointGroupRotation, self).__init__(T=RO, origin=origin)


class PointGroupInversion(PointGroupOperation):
    def __init__(self, origin: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0)):
        """Inversion operation.

        Args:
            origin: origin.
        """
        super(PointGroupInversion, self).__init__(
            T=block_diag(-1 * np.eye(3), 1), origin=origin
        )


class PointGroup:
    def __init__(
        self,
        name: str,
        operations: Sequence[PointGroupOperation],
        ctable: Dict[str, Sequence],
    ):
        """Point group of a molecular or crystal structure.

        Args:
            name: a label for the point group.
            operations: a list of point group operations.
            ctable: character table.
        """
        self.name = name
        assert isinstance(operations, dict)
        for R, op in operations.items():
            assert isinstance(op, PointGroupOperation)
        self.operations = operations
        self.h = len(self.operations)

        assert isinstance(ctable, dict)
        assert all(len(chis) == self.h for chis in ctable.values())
        assert sum(chis[0] ** 2 for chis in ctable.values()) == self.h
        self.ctable = ctable

    def compute_rep_on_orbitals(
        self, orbitals: Sequence[VData], orthogonalize: bool = False
    ) -> Tuple[PointGroupRep, List[str]]:
        """Compute representation matrix on the Hilbert space spanned by a set of orbitals.

        Args:
            orbitals: a set of orbitals.
            orthogonalize: if True, orthorgonalize representation matrix.

        Returns:
            (matrix representation, symmetries)
        """
        rep = PointGroupRep(
            point_group=self, orbitals=orbitals, orthogonalize=orthogonalize
        )

        n = len(orbitals)
        symms = []
        for i, o in enumerate(orbitals):
            vec = np.zeros(n)
            vec[i] = 1
            irprojs = {}
            for irrep, chis in self.ctable.items():
                l = chis[0]
                pvec = np.zeros_like(vec)
                for chi, U in zip(chis, rep.rep_matrices.values()):
                    pvec += chi * U @ vec
                irprojs[irrep] = l / self.h * np.sum(vec * pvec)

            irreps = list(irprojs.keys())
            irproj_values = list(irprojs.values())
            imax = np.argmax(irproj_values)
            # print(irprojs)
            symms.append(f"{irreps[imax]}({irproj_values[imax]:.2f})")

        print("Irrep of orbitals:", symms)
        return rep, symms


class PointGroupRep:
    def __init__(
        self,
        point_group: PointGroup,
        orbitals: Sequence[VData],
        orthogonalize: bool = False,
    ):
        """Representation of a point group on the Hilbert space spanned by a set of orbitals.

        Args:
            point_group: point group.
            orbitals: orbitals spanning the Hilbert space.
            orthogonalize: if True, orthorgonalize representation matrices.
        """
        assert isinstance(point_group, PointGroup)
        self.point_group = point_group

        assert all(isinstance(orbital, VData) for orbital in orbitals)
        cell = orbitals[0].cell
        omega = cell.omega
        N = orbitals[0].nxyz
        self.norb = len(orbitals)

        self.rep_matrices = {
            R: np.zeros([self.norb, self.norb]) for R in self.point_group.operations
        }
        for R, op in self.point_group.operations.items():
            for j in range(self.norb):
                fj = orbitals[j].data
                Rfj = op(fj)
                for i in range(self.norb):
                    fi = orbitals[i].data
                    self.rep_matrices[R][i, j] = omega / N * np.sum(fi * Rfj)

        if orthogonalize:
            # Lowdin orthogonalization
            for R, D in self.rep_matrices.items():
                S = D @ D.T
                U = fractional_matrix_power(S, -1 / 2)
                D[...] = U @ D

        if all(
            [
                np.all(np.isclose(D @ D.T, np.eye(self.norb)))
                for D in self.rep_matrices.values()
            ]
        ):
            print("PointGroupRep: rep matrices are orthogonal")
        else:
            print("PointGroupRep: rep matrices are NOT orthogonal")
