#!/usr/bin/env python3

from koehnlab.molecular import readMolecule, writeMolecule, Molecule

import argparse


def action_orient(molecule: Molecule) -> None:
    molecule.bringToStandardOrientation()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Helper script to process molecules in different ways"
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to the input file containing the molecule's geometry",
        metavar="PATH",
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Path to which the processed molecule shall be written",
        metavar="PATH",
        default=True,
    )
    parser.add_argument(
        "--action",
        help="The action to perform on the molecule",
        choices=["orient"],
        required=True,
    )

    args = parser.parse_args()

    molecule: Molecule = readMolecule(args.input)

    if args.action == "orient":
        action_orient(molecule)
    else:
        raise RuntimeError("Action '%s' not implemented" % args.action)

    writeMolecule(molecule, args.output)


if __name__ == "__main__":
    main()
