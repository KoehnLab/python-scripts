#!/usr/bin/env python3

from typing import List, Tuple, Optional, Dict

import argparse
import numpy as np
import shutil
import subprocess
import os
import re
import io

from koehnlab import finite_differences, utilities, print_utilities


XYZGeom = List[Tuple[str, np.ndarray]]


class ExtractedData:
    def __init__(
        self,
        geometry: XYZGeom,
        magneticMainAxes: np.ndarray,
        gTensor: np.ndarray,
        DTensor: Optional[np.ndarray],
    ):
        self.geometry = geometry
        self.magneticMainAxes = magneticMainAxes
        self.gTensor = gTensor
        self.DTensor = DTensor


class TensorDerivative:
    def __init__(
        self,
        atomIdx: int,
        coordinate: int,
        derivative: np.ndarray,
        coordinateLabels: List[str] = ["x", "y", "z"],
    ):
        self.atomIdx = atomIdx
        self.coordinate = coordinate
        self.derivative = derivative
        self.coordinateLabels = coordinateLabels

    def __str__(self):
        strRep = (
            "# Derivative for distortion of atom number "
            + str(self.atomIdx + 1)
            + " along coordinate "
            + self.coordinateLabels[self.coordinate]
            + " (in angstrom)\n"
        )
        for row in range(len(self.derivative)):
            for col in range(len(self.derivative[row])):
                strRep += "{: 3.8f}  ".format(self.derivative[row][col])

            strRep += "\n"

        return strRep


def runExternal(
    commandLine: List[str], shell: bool = False, requireSuccess: bool = True
) -> Tuple[int, str, str]:
    """Runs the given command line as an external process and returns the tuple (return code, stdout, stdin)
    that results from running this process"""
    process = subprocess.Popen(
        commandLine, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell
    )
    stdout, stderr = process.communicate()
    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")

    if requireSuccess and not process.returncode == 0:
        raise RuntimeError(
            "Executing command line %s was not successful (%d)"
            % (str(commandLine), process.returncode)
        )

    return (process.returncode, stdout, stderr)


def checkMagneticProcessor(exePath: str) -> None:
    """Checks that the magneticProcessor executable at the given path is usable"""
    try:
        retCode, _, stderr = runExternal([exePath, "--help"])
    except Exception as e:
        raise RuntimeError("Invalid path to magneticProcessor '%s': %s" % (exePath, e))

    if retCode != 0:
        raise RuntimeError(
            "Can't use magneticProcessor executable at '%s'%s"
            % (exePath, (": " + stderr) if stderr else "")
        )

    print("Using magneticProcessor at '%s'" % exePath)


def collectOutputFiles(
    baseDir: str, includeRegex: re.Pattern, excludeRegex: Optional[re.Pattern] = None
) -> List[str]:
    """Collects all files under the given base dir that match the includeRegex and don't
    match the excludeRegex (if provided)"""
    outputFiles: List[str] = []

    for path, _, files in os.walk(baseDir):
        for currentFile in files:
            filePath = os.path.join(path, currentFile)

            if re.match(includeRegex, filePath) and (
                excludeRegex is None or not re.match(excludeRegex, filePath)
            ):
                outputFiles.append(filePath)

    return outputFiles


def extractData(filePath: str, magneticProcessorPath: str) -> ExtractedData:
    """Processes the given output file and extracts the relevant information from it (by also running
    the magneticProcessor over it"""
    # Extract geometry from output file
    geometry: XYZGeom = []
    with open(filePath, "r") as outputFile:
        geometrySpec = outputFile.read()
        geometrySpec = geometrySpec[
            geometrySpec.index("ATOMIC COORDINATES")
            + len("ATOMIC COORDINATES") : geometrySpec.index("Bond lengths in Bohr")
        ].strip()
        geometrySpec = geometrySpec[geometrySpec.index("Z") + 1 :].strip()

        # https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
        bohrToAngstrom = 0.529177210903

        for currentLine in geometrySpec.split("\n"):
            # Lines have format <index> <element label> <atomic charge> <x coord> <y coord> <z coord>
            parts = currentLine.split()
            assert len(parts) == 6
            # Convert coordinates to angstrom
            coordinates = [float(x) * bohrToAngstrom for x in parts[3:]]
            geometry.append((parts[1], np.asarray(coordinates)))

    # Now run magneticProcessor on the output and parse its output
    _, processorOutput, _ = runExternal(
        [magneticProcessorPath, "--molpro-file", filePath], requireSuccess=False
    )
    output = io.StringIO(processorOutput)

    # Magnetic main axes
    searchFor = "Magnetic main axes (entries are dimensionless):"
    output.seek(processorOutput.index(searchFor) + len(searchFor) + 1, os.SEEK_SET)
    mainAxes = np.loadtxt(output, max_rows=3)
    assert mainAxes.shape == (3, 3)

    # g-tensor
    searchFor = "g-tensor in the laboratory coordinate system (dimensionless):"
    output.seek(processorOutput.index(searchFor) + len(searchFor) + 1, os.SEEK_SET)
    gTensor = np.loadtxt(output, max_rows=3)
    assert gTensor.shape == (3, 3)

    # D-tensor
    searchFor = "The corresponding D tensor (entries in cm^-1) in original (laboratory) axis system:"
    if searchFor in processorOutput:
        output.seek(processorOutput.index(searchFor) + len(searchFor) + 1, os.SEEK_SET)
        DTensor = np.loadtxt(output, max_rows=3)
        assert gTensor.shape == (3, 3)
    else:
        DTensor = None

    return ExtractedData(
        geometry=geometry, magneticMainAxes=mainAxes, gTensor=gTensor, DTensor=DTensor
    )


def xyzDiff(reference: XYZGeom, compare: XYZGeom) -> np.ndarray:
    """Computes the difference between the given XYZGeom objects"""
    assert len(reference) == len(compare)
    diff = np.zeros(shape=(len(reference), 3))

    for atom in range(len(reference)):
        for coord in range(3):
            diff[atom][coord] = compare[atom][1][coord] - reference[atom][1][coord]

    return diff


def differentiate(
    equilibriumData: ExtractedData, deltas: List[float], dataPoints: List[ExtractedData]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Calculate the derivative of the g- and potentially also D-tensor via finite differences"""
    assert len(deltas) == len(dataPoints)
    assert sorted(deltas) == deltas
    if len(deltas) != 4 or abs(sum(deltas)) > 1e-6:
        raise RuntimeError(
            "For the time being, we only support 4-point central differences"
        )
    # The step in the middle spans over the equilibrium structure and is therefore twice as large
    steps = [deltas[1] - deltas[0], (deltas[2] - deltas[1]) / 2, deltas[3] - deltas[2]]
    for i in range(len(steps) - 1):
        if abs(steps[i + 1] - steps[i]) > 1e-6:
            raise RuntimeError(
                "Expected equidistant steps for the distortions but found " + str(steps)
            )

    step = steps[0]

    gTensorDeriv = np.zeros(shape=(3, 3))
    DTensorDeriv = np.zeros(shape=(3, 3)) if not dataPoints[0].DTensor is None else None

    # Calculate g-tensor element derivatives
    for row in range(len(dataPoints[0].gTensor)):
        for col in range(len(dataPoints[0].gTensor[0])):
            gDiff = finite_differences.central_difference(
                values=[
                    dataPoints[0].gTensor[row][col],
                    dataPoints[1].gTensor[row][col],
                    dataPoints[2].gTensor[row][col],
                    dataPoints[3].gTensor[row][col],
                ],
                delta=step,
            )
            gTensorDeriv[row][col] = gDiff

    if not DTensorDeriv is None:
        # Calculate D-tensor element derivatives
        assert dataPoints[0].DTensor is not None
        assert dataPoints[1].DTensor is not None
        assert dataPoints[2].DTensor is not None
        assert dataPoints[3].DTensor is not None

        for row in range(len(dataPoints[0].DTensor)):
            for col in range(len(dataPoints[0].DTensor[0])):
                DDiff = finite_differences.central_difference(
                    values=[
                        dataPoints[0].DTensor[row][col],
                        dataPoints[1].DTensor[row][col],
                        dataPoints[2].DTensor[row][col],
                        dataPoints[3].DTensor[row][col],
                    ],
                    delta=step,
                )
                DTensorDeriv[row][col] = DDiff

    return (gTensorDeriv, DTensorDeriv)


def calculateDerivatives(
    equilibriumData: ExtractedData, distortedData: List[ExtractedData]
) -> Tuple[List[TensorDerivative], Optional[List[TensorDerivative]]]:
    """Calculates the derivatives for g- and optionally D-tensor based on the provided data"""
    # Step 1: Figure out which data sets belong together and should be used for a single derivative
    differences: List[Tuple[int, int, float]] = []
    for currentData in distortedData:
        diff = xyzDiff(equilibriumData.geometry, currentData.geometry)
        nonZeroEntries = np.argwhere(diff != 0)
        if len(nonZeroEntries) != 1:
            raise RuntimeError(
                "Expected distorted geometries to be elongated along exactly one coordinate"
            )

        atomIdx: int = nonZeroEntries[0][0]
        coordinate: int = nonZeroEntries[0][1]
        delta = diff[atomIdx][coordinate]

        differences.append((atomIdx, coordinate, delta))

    # Sort differences such that the distortions belonging to the same atom are grouped together and within those blocks
    # the distortion along a given coordinate are grouped together as well. However, we have to keep track of the
    # reordering in order to be able to access the correct data object for each distorted geometry.
    dataIndices = list(range(len(distortedData)))
    differences, dataIndices = utilities.compositeSort(differences, dataIndices)

    currentGroupDeltas: List[float] = []
    currentGroupData: List[ExtractedData] = []
    currentGroup: Tuple[int, int] = (differences[0][0], differences[0][0])

    gDerivatives: List[TensorDerivative] = []
    DDerivatives: List[TensorDerivative] = []

    for i in range(len(differences) + 1):
        currentAtomIdx, currentCoordinate, currentDelta = differences[
            min(i, len(differences) - 1)
        ]

        if (currentAtomIdx, currentCoordinate) != currentGroup or i == len(differences):
            # Calculate the derivatives for the entries in the current group
            gDiff, DDiff = differentiate(
                equilibriumData=equilibriumData,
                deltas=currentGroupDeltas,
                dataPoints=currentGroupData,
            )
            gDerivatives.append(
                TensorDerivative(
                    atomIdx=currentGroup[0],
                    coordinate=currentGroup[1],
                    derivative=gDiff,
                )
            )
            if not DDiff is None:
                DDerivatives.append(
                    TensorDerivative(
                        atomIdx=currentGroup[0],
                        coordinate=currentGroup[1],
                        derivative=DDiff,
                    )
                )
            currentGroupData.clear()
            currentGroupDeltas.clear()
            currentGroup = (currentAtomIdx, currentCoordinate)

        if i == len(differences):
            break

        currentGroupDeltas.append(currentDelta)
        currentGroupData.append(distortedData[dataIndices[i]])

    return (gDerivatives, DDerivatives if not len(DDerivatives) == 0 else None)


def transformToBasis(
    derivatives: List[TensorDerivative], basis: np.ndarray
) -> List[TensorDerivative]:
    """Takes the list of derivatives and transforms it into a new coordinate systems where the new
    basis vectors are described in terms of the old coordinates in the columns of basis
    """
    assert basis.shape == (3, 3)

    indices: Dict[int, Dict[int, int]] = {}

    for i in range(len(derivatives)):
        current = derivatives[i]

        if not current.atomIdx in indices:
            indices[current.atomIdx] = {}

        indices[current.atomIdx][current.coordinate] = i

    transformedDerivatives: List[TensorDerivative] = []

    for current in derivatives:
        assert current.coordinate in [0, 1, 2]
        xCoef, yCoef, zCoef = basis[current.coordinate, :]

        xDeriv = derivatives[indices[current.atomIdx][0]].derivative
        yDeriv = derivatives[indices[current.atomIdx][1]].derivative
        zDeriv = derivatives[indices[current.atomIdx][2]].derivative

        transformed = xCoef * xDeriv + yCoef * yDeriv + zCoef * zDeriv
        transformedDerivatives.append(
            TensorDerivative(
                atomIdx=current.atomIdx,
                coordinate=current.coordinate,
                derivative=transformed,
                coordinateLabels=["X", "Y", "Z"],
            )
        )

    return transformedDerivatives


def writeToFile(
    derivatives: List[TensorDerivative], path: str, comment: Optional[str]
) -> None:
    """Writes the given derivatives to a file"""
    with open(path, "w") as outputFile:
        if comment:
            print(
                "# " + comment,
                file=outputFile,
            )
            print("", file=outputFile)

        for current in derivatives:
            print(current, file=outputFile)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Computes the derivatives of magnetic properties (g- and D-tensor) "
        + "based on single_aniso calculations for a set of distorted geometries"
    )
    parser.add_argument(
        "--equilibrium-calc",
        help="Path to the output file for the calculation on the equilibrium structure",
        metavar="PATH",
        required=True,
    )
    parser.add_argument(
        "--distorted-calc-dir",
        help="Path to the directory in which the output files for the calculations on the distorted geometries are located",
        metavar="PATH",
        default=".",
    )
    parser.add_argument(
        "--include",
        help="A regular expression used for matching valid paths of output files that shall be considered",
        metavar="REGEX",
        default=".+\\.out$",
    )
    parser.add_argument(
        "--exclude",
        help="A regular expression used for excluding matching output files from the processing",
        metavar="REGEX",
        default=None,
    )
    parser.add_argument(
        "--magnetic-processor",
        help="Path to the magneticProcessor executable (https://github.com/KoehnLab/magneticProcessor)",
        metavar="PATH",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        help="Path to the directory into which the output files shall be written",
        metavar="PATH",
        default=".",
    )

    args = parser.parse_args()

    # Check magneticProcessor is accessible
    if not args.magnetic_processor:
        # Search in PATH
        args.magnetic_processor = shutil.which("magneticProcessor")
    if not args.magnetic_processor:
        raise RuntimeError(
            "'magneticProcessor' is not in PATH and you did not provide the --magnetic-processor option"
        )
    checkMagneticProcessor(args.magnetic_processor)

    # Check provided paths
    if not os.path.isfile(args.equilibrium_calc):
        raise RuntimeError(
            "Equilibrium output file '%s' doesn't exist or is not a file"
            % args.equilibrium_calc
        )
    if not os.path.isdir(args.distorted_calc_dir):
        raise RuntimeError(
            "Directory containing calculations for distorted geometries '%s' doesn't exist or is not a directory"
            % args.distorted_calc_dir
        )

    print(
        "Recursively searching for calculations on distorted geometries in '%s'"
        % args.distorted_calc_dir
    )

    # Find all relevant output files
    distortedCalcOutputFiles = collectOutputFiles(
        args.distorted_calc_dir,
        includeRegex=re.compile(args.include),
        excludeRegex=re.compile(args.exclude) if args.exclude else None,
    )

    print(
        "Processing %d calculations on distorted geometries"
        % len(distortedCalcOutputFiles)
    )

    # Fetching all necessary data
    equilibriumData = extractData(
        args.equilibrium_calc, magneticProcessorPath=args.magnetic_processor
    )
    distortedData: List[ExtractedData] = []
    for currentOutput in utilities.progressbar(
        distortedCalcOutputFiles, prefix="Extracting data "
    ):
        distortedData.append(
            extractData(currentOutput, magneticProcessorPath=args.magnetic_processor)
        )

    # Calculate derivatives
    gTensorDerivatives, DTensorDerivatives = calculateDerivatives(
        equilibriumData, distortedData
    )

    laboratoryToMainAxes = np.linalg.inv(equilibriumData.magneticMainAxes)

    # Write out the calculated derivates and also the ones transformed into the system of magnetic main axes (of the equilibrium geometry)
    writeToFile(
        gTensorDerivatives,
        path=os.path.join(args.output_dir, "g-tensor_derivatives"),
        comment="First derivative of the g-tensor elements w.r.t. changes in coordinates (laboratory coordinate system) of the atoms",
    )

    gTensorDerivatives = transformToBasis(
        gTensorDerivatives, basis=laboratoryToMainAxes
    )
    writeToFile(
        gTensorDerivatives,
        path=os.path.join(args.output_dir, "g-tensor_derivatives_main_axes"),
        comment="First derivative of the g-tensor elements w.r.t. changes in coordinates (coordinate system of magnetic main axes) of the atoms",
    )

    if not DTensorDerivatives is None:
        writeToFile(
            DTensorDerivatives,
            path=os.path.join(args.output_dir, "D-tensor_derivatives"),
            comment="First derivative of the D-tensor elements w.r.t. changes in coordinates (laboratory coordinate system) of the atoms",
        )

        DTensorDerivatives = transformToBasis(
            DTensorDerivatives, basis=laboratoryToMainAxes
        )
        writeToFile(
            DTensorDerivatives,
            path=os.path.join(args.output_dir, "D-tensor_derivatives_main_axes"),
            comment="First derivative of the D-tensor elements w.r.t. changes in coordinates (coordinate system of magnetic main axes) of the atoms",
        )

    # Also write out magnetic main axes
    with open(os.path.join(args.output_dir, "magnetic_main_axes"), "w") as outputFile:
        print(
            "# Magnetic main axes (column-wise) of the undistorted molecule\n"
            + "# (the matrix describes the transformation from laboratory to main axes coordinates)",
            file=outputFile,
        )
        print_utilities.printMat(
            equilibriumData.magneticMainAxes, file=outputFile, useLabels=False
        )


if __name__ == "__main__":
    main()
