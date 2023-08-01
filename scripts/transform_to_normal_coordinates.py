#!/usr/bin/env python3

from typing import List
from numpy.typing import NDArray

import argparse

import numpy as np


def get_atomic_coordinates(npzFile) -> NDArray:
    if not "atomic_coordinates" in npzFile and "at_pos" in npzFile:
        # Compatibility with old version of extraction script
        return npzFile["at_pos"]

    return npzFile["atomic_coordinates"]


def get_element_symbols(npzFile) -> List[str]:
    if not "element_symbols" in npzFile and "at_types" in npzFile:
        # Compatibility with old version of extraction script
        return npzFile["at_types"]

    return npzFile["element_symbols"]


def get_atomic_masses(npzFile) -> NDArray:
    if not "atomic_masses" in npzFile and "at_mass" in npzFile:
        # Compatibility with old version of extraction script
        return npzFile["at_mass"]

    return npzFile["atomic_masses"]


def get_frequencies(npzFile) -> NDArray:
    if not "frequencies" in npzFile and "freq" in npzFile:
        # Compatibility with old version of extraction script
        return npzFile["freq"]

    return npzFile["frequencies"]


def get_normal_coordinates(npzFile) -> NDArray:
    if not "normal_coordinates" in npzFile and "Lmat" in npzFile:
        # Compatibility with old version of extraction script
        return npzFile["Lmat"]

    return npzFile["normal_coordinates"]


def compute_prefactor_matrix(nAtoms: int, masses: NDArray, frequencies: NDArray, zero_out_dofs: bool = True) -> NDArray:
    # Incorporate additional pre-factors into Lmat at this point to only do it once
    # The underlying formula can be found in e.g. https://doi.org/10.1126/sciadv.aax7163 (eq. 12)
    prefactor_matrix = np.ones(shape=(3 * nAtoms, 3 * nAtoms))

    # First: Undo atom-wise mass-weighting
    for atom_idx in range(nAtoms):
        inv_mass = 1 / np.sqrt(masses[atom_idx])
        prefactor_matrix[3 * atom_idx + 0, :] *= inv_mass
        prefactor_matrix[3 * atom_idx + 1, :] *= inv_mass
        prefactor_matrix[3 * atom_idx + 2, :] *= inv_mass


    # In order for the units to work out, we have to apply a factor. This factor assumes
    # that the frequencies have been given in inverse centimeters and the displacements
    # were in Angstroms.
    conv_factor = 1 / 0.17222125665281293 #  Å / cm^{-1/2}

    # Second: Perform mode-wise frequency weighting
    for mode_idx in range(3 * nAtoms):
        if frequencies[mode_idx] < 1E-3:
            if zero_out_dofs:
                # Zero out translational and rotational DOFs
                prefactor_matrix[:, mode_idx] *= 0
        else:
            inv_freq = 1 / np.sqrt(frequencies[mode_idx])
            prefactor_matrix[:, mode_idx] *= inv_freq * conv_factor

    return prefactor_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Transforms a given derivative(-like) properties per distortions along cartesian coordinates to derivative(-like) "
        + "properties per distortions along normal coordinates"
    )
    parser.add_argument(
        "--freq-info",
        required=True,
        metavar="PATH",
        help="Path to the .npz file containing information about the performed harmonic frequency calculation",
    )
    parser.add_argument(
        "--derivatives",
        required=True,
        metavar="PATH",
        help="Path to the .npz file containing the derivative(s) that shall be transformed",
    )
    parser.add_argument(
        "--drift-correction",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Whether to apply drift-corrections",
    )
    parser.add_argument("--output-file", default="transformed_derivatives.npz", metavar="PATH", help="Path to where the result shall be written")
    parser.add_argument("--zero-out-dofs", type=bool, default=True, action=argparse.BooleanOptionalAction, help="Whether to zero-out the derivatives for translational and rotational DOFs")

    args = parser.parse_args()

    freq_info = np.load(args.freq_info)
    derivatives = np.load(args.derivatives)

    Lmat = get_normal_coordinates(freq_info)
    masses = get_atomic_masses(freq_info)
    frequencies = get_frequencies(freq_info)
    nAtoms = len(masses)

    assert Lmat.shape == (3 * nAtoms, 3 * nAtoms)
    assert len(frequencies) == 3 * nAtoms


    # This is an element-wise multiplication (Hadamard product)!
    Lmat *= compute_prefactor_matrix(nAtoms=nAtoms, masses=masses, frequencies=frequencies, zero_out_dofs=args.zero_out_dofs)


    # Now turn to actually transforming derivatives
    transformed_derivatives = {}
    for current_derivative_name in derivatives.keys():
        print("Transforming {}...".format(current_derivative_name))
        current_derivative = derivatives[current_derivative_name]

        assert len(current_derivative) == 3 * nAtoms

        # Drift calculation (+ correction)
        # Slicing syntax: [start:stop:step]
        xDrift = np.sum(current_derivative[0::3], axis=0) / nAtoms
        yDrift = np.sum(current_derivative[1::3], axis=0) / nAtoms
        zDrift = np.sum(current_derivative[2::3], axis=0) / nAtoms

        print("X drift per atom:\n", xDrift)
        print("Y drift per atom:\n", yDrift)
        print("Z drift per atom:\n", zDrift)

        if args.drift_correction:
            print("-> correcting drift")
            current_derivative[0::3] -= xDrift
            current_derivative[1::3] -= yDrift
            current_derivative[2::3] -= zDrift

        # Now, actually perform the transformation
        transformed = np.zeros(shape=np.shape(current_derivative))
        for mode_idx in range(3 * nAtoms):
            for atom_coord in range(3 * nAtoms):
                transformed[mode_idx] += current_derivative[atom_coord] * Lmat[atom_coord,mode_idx]

        transformed_derivatives[current_derivative_name] = transformed
        print()

    np.savez_compressed(args.output_file, **transformed_derivatives)
    print("Saved transformed derivatives to '{}'".format(args.output_file))



if __name__ == "__main__":
    main()
