#!/usr/bin/env python3

from typing import List
from numpy.typing import NDArray

import argparse

from ase import Atoms
from ase.io.turbomole import write_turbomole

import numpy as np

# Hartree in cm^-1   Eh/(h*c*100)
Eh_rcm = 219474.631330
# atomic mass units in multiples of electron masses
amu_me = 1822.8884862086924302
# bohr radius in Angstrom
bohr_ang = 0.529177210903 

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



def reweight(Lmat,masses,freqs):
    ''' apply sqrt(hbar/(m omega)) factor, using atomic units '''

    nAtoms = len(masses)
    assert len(freqs) == 3*nAtoms

    # apply mass weight to rows
    massw = np.zeros((3*nAtoms))
    jdx = 0
    for idx in range(nAtoms):
        sqmass = 1.0/np.sqrt(masses[idx])
        massw[jdx+0] = sqmass
        massw[jdx+1] = sqmass
        massw[jdx+2] = sqmass
        jdx += 3

    #for idx in range(3*nAtoms):
    #    Lmat[idx,:] *= massw

    # apply freq weight to columns
    freqw = np.zeros((3*nAtoms))

    for idx in range(3*nAtoms):
        if freqs[idx] > 1e-8:
            freqw[idx] = 1.0/np.sqrt(freqs[idx])

    #for idx in range(3*nAtoms):
    #    Lmat[:,idx] *= freqw
    for idx in range(3*nAtoms):
        for jdx in range(3*nAtoms):
            Lmat[idx,jdx] *= massw[idx]*freqw[jdx]

    return Lmat

def main():
    parser = argparse.ArgumentParser(
        description="Generate a grid of unit normal mode displacements "
    )
    parser.add_argument(
        "--freq-info",
        required=True,
        metavar="PATH",
        help="Path to the .npz file containing information about the performed harmonic frequency calculation",
    )
    parser.add_argument(
        "--order",
        default=1,
        metavar="VALUE",
        help="order of derivative (supported: 1, 2)",
    )
    parser.add_argument(
        "--stencil",
        default=2,
        metavar="VALUE",
        help="order of grid (supported: 1, 2)",
    )
    parser.add_argument(
        "--increment",
        default=0.1,
        metavar="VALUE",
        help="increment for dimensionless displacements",
    )
    parser.add_argument(
        "--min-wavenumber","-L",
        default=5,
        metavar="VALUE",
        help="minimum wavenumber of considered vibrations",
    )
    parser.add_argument(
        "--max-wavenumber","-H",
        default=500,
        metavar="VALUE",
        help="maximum wavenumber of considered vibrations",
    )
    parser.add_argument("--output-file", default="coord", metavar="PATH", help="Path to where the result shall be written")

    args = parser.parse_args()

    freq_info = np.load(args.freq_info)

    output_file = args.output_file

    stencil = int(args.stencil)
    order = int(args.order)

    Lmat = get_normal_coordinates(freq_info)
    masses = get_atomic_masses(freq_info)*amu_me  # convert to atomic units
    frequencies = get_frequencies(freq_info)/Eh_rcm  # convert to atomic units
    nAtoms = len(masses)

    coordinates = get_atomic_coordinates(freq_info)/bohr_ang  # convert to atomic units
    atom_names = get_element_symbols(freq_info)

    minfreq = float(args.min_wavenumber)/Eh_rcm
    maxfreq = float(args.max_wavenumber)/Eh_rcm

    inc = float(args.increment)

    assert Lmat.shape == (3 * nAtoms, 3 * nAtoms)
    assert len(frequencies) == 3 * nAtoms

    Lmat = reweight(Lmat,masses,frequencies)

    # write out reference coordinates
    file_name = output_file+"_0"
    with open(file_name,'w') as outfile:
        # ase expects angstrom units :/
        write_turbomole(outfile,Atoms(symbols=atom_names,positions=coordinates*bohr_ang))

    
    for idx in range(3 * nAtoms):

        freq = frequencies[idx]
        if freq < minfreq or freq > maxfreq:
            continue

        for dsp in range(-stencil,stencil+1):
            if dsp == 0:
                continue

            coord_dist = coordinates + float(dsp)*inc*Lmat[:,idx].reshape((nAtoms,3))

            if dsp > 0:
                dsp_str = f"p{dsp:1d}" 
            else:
                dsp_str = f"m{-dsp:1d}"
            file_name = output_file+f"_1_{idx+1:0>3d}_"+dsp_str
            with open(file_name,'w') as outfile:
                write_turbomole(outfile,Atoms(symbols=atom_names,positions=coord_dist*bohr_ang))
 
            if order < 2:
                continue

            for jdx in range(idx+1, 3 * nAtoms):
                
                freq2 = frequencies[jdx]
                if freq2 < minfreq or freq2 > maxfreq:
                    continue

                for dsp2 in range(-stencil,stencil+1):
                    if dsp2 == 0:
                        continue

                    coord_dist2 = coord_dist + float(dsp2)*inc*Lmat[:,jdx].reshape((nAtoms,3))

                    if dsp2 > 0:
                        dsp2_str = f"p{dsp2:1d}"
                    else:
                        dsp2_str = f"m{-dsp2:1d}"
                    file_name = output_file+f"_2_{idx+1:0>3d}_{jdx+1:0>3d}_"+dsp_str+"_"+dsp2_str
                    with open(file_name,'w') as outfile:
                        write_turbomole(outfile,Atoms(symbols=atom_names,positions=coord_dist2*bohr_ang))


if __name__ == "__main__":
    main()
