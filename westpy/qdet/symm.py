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

from westpy import VData


class PointGroupOperation:
    def __init__(
        self,
        T: np.ndarray,
        origin: Union[Sequence[float], np.ndarray] = None,
        cell: np.ndarray = None,
    ):
        """An operation in the point group.

        Args:
            T: 4x4 affine transformation matrix for point group operation
            origin: origin of operation, defined as follows:
                  1. [x,y,z] = coordinate of origin in crystal unit
                  2. [nx,ny,nz] = shape of the volumetric data, i.e.,
                     dimension of the real space grid
                  3. origin = [x*nx,y*ny,z*nz]
            cell: transformation matrix to the volumetric cell data, defined
                  as follows:
                  1. A = crystal vector matrix
                     np.array([x1,y1,z1],
                              [x2,y2,z2],
                              [x3,y3,z3]).T
                     where [xi,yi,zi] is the ith crystal basis vector in any unit
                  2. B = 1 / np.array([nx,0,0],
                              [0,ny,0],
                              [0,0,nz])
                     where nx,ny,nz are the shape of the volumetric data, i.e.,
                     dimension of the real space grid
                  3. cell = A @ B
        """
        assert T.shape == (4, 4)
        self.T = T
        if cell is not None:
            assert np.shape(cell) == (3, 3)
            self.set_coord(cell)
        if origin is not None:
            assert len(origin) == 3
            self.set_origin(origin)

    def set_coord(self, cell):
        """transform the coordinate of T into M^-1M"""
        TC = block_diag(cell, 1)
        self.T = np.linalg.inv(TC) @ self.T @ TC

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
        return affine_transform(f, matrix=self.T, mode="grid-wrap")


class PointGroupReflection(PointGroupOperation):
    def __init__(
        self,
        normal: Union[Sequence[float], np.ndarray],
        origin: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
        cell: np.ndarray = None,
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
        super(PointGroupReflection, self).__init__(T=RE, origin=origin, cell=cell)


class PointGroupRotation(PointGroupOperation):
    def __init__(
        self,
        rotvec: Union[Sequence[float], np.ndarray],
        origin: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
        cell: np.ndarray = None,
    ):
        """Rotation operation.

        reflection matrix RO

        Args:
            normal: rotvec: (a, b, c), |(a, b, c)| is interpreted as degree in radian, direction is interpreted as axis
            origin: origin
        """
        rotation = Rotation.from_rotvec(rotvec)
        RO = block_diag(rotation.as_matrix().T, 1)
        super(PointGroupRotation, self).__init__(T=RO, origin=origin, cell=cell)


class PointGroupInversion(PointGroupOperation):
    def __init__(
        self,
        origin: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
        cell: np.ndarray = None,
    ):
        """Inversion operation.

        Args:
            origin: origin.
        """
        super(PointGroupInversion, self).__init__(
            T=block_diag(-1 * np.eye(3), 1), origin=origin, cell=cell
        )


class PointGroupRotateReflection(PointGroupOperation):
    def __init__(
        self,
        rotvec: Union[Sequence[float], np.ndarray],
        origin: Union[Sequence[float], np.ndarray] = (0.0, 0.0, 0.0),
        multiple: int = 1,
        cell: np.ndarray = None,
    ):
        """Rotation and then reflect in a plane perpendicular to the rotation axis

        Rotation matrix R0
        reflection matrix RE
        Rotation+reflection=R1@R0

        Args:
            normal: rotvec: (a, b, c), |(a, b, c)| is interpreted as degree in radian, direction is interpreted as axis
            origin: origin
            multiple is an integer. give the mutiplication of the operation: if multiple=3 then give S^3=S@S@S  operation
        """
        # print('Rotvec =',rotvec)
        R0 = Rotation.from_rotvec(rotvec).as_matrix()
        # print('R0 is',R0)
        a, b, c = np.array(rotvec) / np.linalg.norm(rotvec)
        RE = np.array(
            [
                [1 - 2 * a * a, -2 * a * b, -2 * a * c],
                [-2 * a * b, 1 - 2 * b * b, -2 * b * c],
                [-2 * a * c, -2 * b * c, 1 - 2 * c * c],
            ]
        )
        # print(type(RE),type(R0))
        S = RE @ R0  # 3x3 matrix for rotation then reflection.
        if multiple > 1:  # if need multiple S operation
            for m in range(1, multiple):
                S = RE @ R0 @ S
        # print(S)
        S_Affine = block_diag(S.T, 1)  # construct affine matrix
        # print(S_Affine,type(S_Affine))
        super(PointGroupRotateReflection, self).__init__(
            T=S_Affine, origin=origin, cell=cell
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
