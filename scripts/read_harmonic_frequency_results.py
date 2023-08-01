#!/usr/bin/env python3

from typing import List, Tuple
from numpy.typing import NDArray

import argparse
import os

import numpy as np

from ase.io.turbomole import read_turbomole


def process_turbomole(
    directory: str,
) -> Tuple[List[str], NDArray, NDArray, NDArray, NDArray]:
    """Extracts information of a harmonic frequency calculation performed via TurboMole"""
    if not os.path.exists(directory):
        raise RuntimeError("Directory '{}' does not exist".format(directory))
    if not os.path.isdir(directory):
        raise RuntimeError(
            "Provided file ('{}') is not a directory, but for TurboMole a directory was expected".format(
                directory
            )
        )

    # Read coordinates and masses
    with open(os.path.join(directory, "coord"), "r") as coord_file:
        atoms = read_turbomole(coord_file)

    coordinates: NDArray = atoms.get_positions()
    masses: NDArray = atoms.get_masses()
    elements: List[str] = atoms.get_chemical_symbols()
    nAtoms: int = atoms.get_global_number_of_atoms()

    assert len(coordinates) == nAtoms
    assert len(masses) == nAtoms

    # Read frequencies
    with open(os.path.join(directory, "vibspectrum"), "r") as vib_spec_file:
        lines: List[str] = vib_spec_file.readlines()
        assert len(lines) > 1 and lines[0].startswith(
            "$vibrational spectrum"
        ), "vibspectrum file does not have expected format"

        frequencies: NDArray = np.zeros(3 * nAtoms)
        freq_idx = 0
        for current_line in lines[1:]:
            if current_line.strip().startswith("#") or current_line.strip().startswith(
                "$"
            ):
                continue

            parts = current_line.split()
            nParts = len(parts)
            assert nParts in [5, 6]

            if nParts == 5:
                # Translational and rotational DOFs
                frequencies[freq_idx] = float(parts[1])
                freq_idx += 1
            elif nParts == 6:
                # Regular (proper) frequency
                frequencies[freq_idx] = float(parts[2])
                freq_idx += 1

    # Read normal modes
    with open(os.path.join(directory, "vib_normal_modes"), "r") as vib_mode_file:
        lines: List[str] = vib_mode_file.readlines()
        assert len(lines) > 1 and lines[0].startswith(
            "$vibrational normal modes"
        ), "vib_normal_modes file has unexpected format"

        normal_mode_entries: List[float] = []

        for current_line in lines[1:]:
            if current_line.strip().startswith("#") or current_line.strip().startswith(
                "$"
            ):
                continue
            # Ignore element indices at the beginning of the line
            parts: List[str] = current_line[5:].split()

            normal_mode_entries.extend([float(x) for x in parts])

        # We expect 3N entries per normal mode (x,y,z coordinates for every atom) and in
        # total there should be 3N normal modes, as we are also including the translational
        # and rotational DOFs
        assert len(normal_mode_entries) == 9 * nAtoms * nAtoms

        # Re-assemble the normal coordinates as a 3N x 3N matrix
        Lmat = np.reshape(np.array(normal_mode_entries), (3 * nAtoms, 3 * nAtoms))

        # Re-introduce mass weighting
        for mode in range(3 * nAtoms):
            for row in range(3 * nAtoms):
                atom: int = row // 3
                Lmat[row, mode] *= np.sqrt(masses[atom])

        # Re-normalize
        for mode in range(3 * nAtoms):
            norm = np.linalg.norm(Lmat[:, mode])
            Lmat[:, mode] /= norm

        # Safety check
        product: NDArray = np.matmul(Lmat, Lmat.T)
        assert (
            np.sum(np.abs(product - np.diag(product.diagonal())) > 1e-4) == 0
        ), "Expected Lmat * Lmat^T to be diagonal, but wasn't"

        return (elements, coordinates, masses, frequencies, Lmat)


def main():
    parser = argparse.ArgumentParser(
        description="Read results from a harmonic frequency calculation and store them in a Numpy .npz file"
    )
    parser.add_argument(
        "--turbomole-dir",
        default=None,
        help="The directory in which the TurboMole results of the calculation are stored",
        metavar="PATH",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default="harmonic_frequency_results.npz",
        help="File to which the output shall be written",
        metavar="PATH",
    )

    args = parser.parse_args()

    if args.turbomole_dir is None:
        raise RuntimeError(
            "--turbomole-dir option not specified, but no other backends are implemented yet"
        )

    if not args.turbomole_dir is None:
        elements, coordinates, masses, frequencies, Lmat = process_turbomole(
            args.turbomole_dir
        )
        program = "TurboMole"
    else:
        raise RuntimeError("No input specified")

    np.savez_compressed(
        args.output_file,
        atomic_coordinates=coordinates,
        frequencies=frequencies,
        normal_coordinates=Lmat,
        extracted_from=program,
        atomic_masses=masses,
        element_symbols=elements,
    )
    print("Saved extracted data to {}".format(args.output_file))


if __name__ == "__main__":
    main()
