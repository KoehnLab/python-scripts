from enum import Enum

from .Molecule import Molecule
from .Atom import Atom
from .Element import Element

import numpy as np


class FileFormat(Enum):
    XYZ = 0


def readMolecule(path: str, fmt: FileFormat = FileFormat.XYZ) -> Molecule:
    """Reads a molecule in the given format from the given path on the filesystem"""
    if fmt != FileFormat.XYZ:
        raise RuntimeError("Unsupported file format %s" % str(fmt))

    molecule = Molecule()
    with open(path, "r") as inputFile:
        lines = inputFile.readlines()

        if len(lines) < 3:
            raise RuntimeError("Invalid XYZ file '%s'" % path)

        nAtoms = int(lines[0])

        # second line is only the comment, which we don't care about

        for i in range(nAtoms):
            components = lines[i + 2].split()

            if len(components) < 4:
                raise RuntimeError("Invalid line in XYZ file: '%s':%d" % (path, i + 2))

            molecule.addAtom(
                Atom(
                    element=Element[components[0]],
                    coordinates=np.asarray(
                        [
                            float(components[1]),
                            float(components[2]),
                            float(components[3]),
                        ]
                    ),
                )
            )

    assert len(molecule.atoms) == nAtoms

    return molecule


def writeMolecule(
    molecule: Molecule, path: str, fmt: FileFormat = FileFormat.XYZ
) -> None:
    """Writes the given Molecule's geometry to a file at the given path and in the specified format"""
    if fmt != FileFormat.XYZ:
        raise RuntimeError("Unsupported file format %s" % str(fmt))

    with open(path, "w") as outputFile:
        outputFile.write("{}\n\n".format(len(molecule.atoms)))
        for currentAtom in molecule.atoms:
            outputFile.write(
                "{: <2s} {: 17.12f} {: 17.12f} {: 17.12f}\n".format(
                    currentAtom.element.symbol(),
                    currentAtom.coordinates[0],
                    currentAtom.coordinates[1],
                    currentAtom.coordinates[2],
                )
            )
