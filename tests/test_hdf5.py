#!/usr/bin/env python3

import unittest
import os

from koehnlab.io import get_property_matrix, get_soc_matrix, get_state_meta, Basis, transform_to_product_basis, MatrixType, similarity_transform
from koehnlab.spin_hamiltonians import spinMat, spin_mat, Coordinate3D, compute_magnetic_moment_matrix, compute_A_matrix, compute_g_tensor
from koehnlab.print_utilities import printMat

import numpy as np

script_dir: str = os.path.dirname(os.path.realpath(__file__))
data_dir: str = os.path.join(script_dir, "data")
hdf5_file: str = os.path.join(data_dir, "soci_data.hdf5")


class TestMolproHDF5(unittest.TestCase):
    def test_state_meta(self):
        meta = get_state_meta(hdf5_file)

        np.testing.assert_equal(meta.counts, [4, 2, 2, 2])
        np.testing.assert_equal(meta.irreps, [1, 2, 3, 4])
        np.testing.assert_equal(meta.spin_qns, [1, 1, 1, 1])

    def test_property_matrices(self):
        for prop_name in ["DMX", "DMY", "DMZ", "LX(RH)", "LY(RH)", "LZ(RH)"]:
            for basis in [Basis.WF0, Basis.SpinOrbit]:
                with self.subTest(property=prop_name, basis=basis):
                    actual = get_property_matrix(hdf5_file, prop=prop_name, basis=basis)

                    filename = prop_name.lower()
                    if "(" in filename:
                        filename = filename[: filename.find("(")]

                    if basis == Basis.SpinOrbit:
                        expected = 1j * np.loadtxt(
                            os.path.join(
                                data_dir,
                                "soci_data_" + filename + "_transformed_imag.csv",
                            ),
                            delimiter=",",
                        )
                        expected += np.loadtxt(
                            os.path.join(
                                data_dir,
                                "soci_data_" + filename + "_transformed_real.csv",
                            ),
                            delimiter=",",
                        )
                    else:
                        expected = np.loadtxt(
                            os.path.join(data_dir, "soci_data_" + filename + ".csv"),
                            delimiter=",",
                        )

                        if prop_name.startswith("L"):
                            expected = expected * 1j

                    if basis == Basis.SpinOrbit:
                        # Eigenvectors have an undetermined phase. So we have to check the absolute values
                        # in order to not get false negatives due to phase differences
                        np.testing.assert_almost_equal(np.abs(actual), np.abs(expected), decimal=6)  # type: ignore
                    else:
                        np.testing.assert_almost_equal(actual, expected, decimal=6)  # type: ignore

    def test_soc_energies(self):
        soc_mat = get_soc_matrix(hdf5_file)

        # SOC mat must be hermitian
        np.testing.assert_almost_equal(soc_mat, np.conj(soc_mat.T))

        energies, _ = np.linalg.eigh(soc_mat)

        expected = np.loadtxt(
            os.path.join(data_dir, "soci_data_energies.csv"), delimiter=","
        )

        np.testing.assert_almost_equal(energies, expected)

    def test_g_tensor(self):
        pseudospin = 1
        pseudomult = int(2 * pseudospin) + 1

        meta = get_state_meta(hdf5_file)
        Lx = get_property_matrix(hdf5_file, prop="LX(RH)", basis=Basis.SpinOrbit)
        Ly = get_property_matrix(hdf5_file, prop="LY(RH)", basis=Basis.SpinOrbit)
        Lz = get_property_matrix(hdf5_file, prop="LZ(RH)", basis=Basis.SpinOrbit)

        Sx = transform_to_product_basis(spin_mat(spin_qns=meta.spin_qns, component=Coordinate3D.X), spin_qns=meta.spin_qns, state_nums=meta.counts, matrix_type=MatrixType.Spin)
        Sy = transform_to_product_basis(spin_mat(spin_qns=meta.spin_qns, component=Coordinate3D.Y), spin_qns=meta.spin_qns, state_nums=meta.counts, matrix_type=MatrixType.Spin)
        Sz = transform_to_product_basis(spin_mat(spin_qns=meta.spin_qns, component=Coordinate3D.Z), spin_qns=meta.spin_qns, state_nums=meta.counts, matrix_type=MatrixType.Spin)

        soc_mat = get_soc_matrix(hdf5_file)
        _, eigvecs = np.linalg.eigh(soc_mat)

        Sx = similarity_transform(Sx, eigvecs)
        Sy = similarity_transform(Sy, eigvecs)
        Sz = similarity_transform(Sz, eigvecs)

        muX = compute_magnetic_moment_matrix(Sx, Lx, in_bohr_magnetons=True)
        muY = compute_magnetic_moment_matrix(Sy, Ly, in_bohr_magnetons=True)
        muZ = compute_magnetic_moment_matrix(Sz, Lz, in_bohr_magnetons=True)

        A = compute_A_matrix(muX, muY, muZ, pseudomult)

        g_values, _ = compute_g_tensor(A, S=pseudospin, in_bohr_magnetons=True)

        g_values.sort()

        np.testing.assert_almost_equal(g_values, [2.001931066, 2.021692806, 2.022017269]) 


if __name__ == "__main__":
    unittest.main()
