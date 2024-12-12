from dataclasses import dataclass, field

from .basis_util import transform_to_product_basis, similarity_transform
from .matrix_type import MatrixType
from .basis import Basis

import numpy as np
from numpy.typing import NDArray

import h5py


def convert_soc_mat(matrix_3d):
    """
    Converts the 3D representation of the complex SOC matrix (first dimension
    switching between real and imaginary part) to a proper complex-valued
    2D matrix representation.
    """
    return 1j * matrix_3d[1, :, :] + matrix_3d[0, :, :]


def get_soc_matrix(file: str | h5py.File):
    """
    Extracts the Spin-Orbit-Coupling matrix from the given HDF5 file

    Args:
    ----------------------
    file -- Either a HDF5 file handle or the path to the HDF5 file on disk
    Returns:
    ----------------------
    The SOC matrix
    """

    if type(file) is h5py.File:
        return convert_soc_mat(file["SOC matrix"])

    with h5py.File(file, "r") as h5file:
        return convert_soc_mat(h5file["SOC matrix"])


def get_property_matrix(path: str, prop: str, basis: Basis):
    """
    Returns the matrix of a given operator in a given basis which is stored
    in a HDF5 File in the given path.

    Args:
    ----------------------
    path -- The path the HDF5 file is stored
    prop -- the operator of which the matrix is extracted from Molpro.
            (DMX,DMY,DMZ,LX(),LY(),LZ(),SOC matrix)
    basis -- the basis in which the property matrix should be returned.
             Choose one of the three options of Basis Enum
    Returns:
    ---------------------
    PropMat -- The property matrix of given operator in the given basis

    """
    with h5py.File(path, "r") as h5file:
        spin_qns = np.array(h5file["Spin QNs"][:])  # type: ignore
        num_states = np.array(h5file["Spatial states"][:])  # type: ignore

        if "Description" in h5file[prop].attrs and h5file[prop].attrs[
            "Description"
            ].lower().startswith("imaginary part of"): # type: ignore
            prop_mat = 1j * h5file[prop][:]  # type: ignore
        else:
            prop_mat = h5file[prop][:]  # type: ignore

        if basis == Basis.WF0:
            return prop_mat
        elif basis == Basis.Product:
            return transform_to_product_basis(
                prop_mat, spin_qns, num_states, MatrixType.Spatial
            )
        elif basis == Basis.SpinOrbit:
            product_mat = transform_to_product_basis(
                prop_mat, spin_qns, num_states, MatrixType.Spatial
            )

            soc_mat = get_soc_matrix(h5file)
            _, eigvecsh = np.linalg.eigh(soc_mat)

            return similarity_transform(product_mat, eigvecsh)

        raise Exception(f'Unsupported basis value: "{basis.name}"')


@dataclass
class StateMeta:
    irreps: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(shape=1, dtype=np.intp)
    )
    counts: NDArray[np.intp] = field(
        default_factory=lambda: np.empty(shape=1, dtype=np.intp)
    )
    spin_qns: NDArray[np.float64] = field(
        default_factory=lambda: np.empty(shape=1, dtype=np.float64)
    )


def get_state_meta(path: str) -> StateMeta:
    """
    Returns meta information about the states contained in the Molpro HDF5 dump

    Args:
    -----------------------
    path -- the path to the HDF5 File where the arrays from Molpro are stored

    Returns:
    -----------------------
    The meta information
    """
    meta = StateMeta()
    with h5py.File(path, "r") as h5file:
        meta.spin_qns = np.array(h5file["Spin QNs"][:])  # type: ignore
        meta.counts = np.array(h5file["Spatial states"][:])  # type: ignore
        meta.irreps = np.array(h5file["IRREPs"][:])  # type: ignore

    return meta
