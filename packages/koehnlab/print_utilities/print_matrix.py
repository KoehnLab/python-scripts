# a few convenient output routines

from typing import Optional, Type, Any, TextIO

import math
import numpy as np
import sys


def getFormatWidth(fmt: str) -> int:
    """Gets the field width of formatting a number with the given formatting string"""
    return len(fmt.format(1))


def printMat(
    matrix,
    entryFmt: Optional[str] = None,
    colsPerLine: int = 10,
    useLabels: bool = True,
    triangle: bool = False,
    dtype: Optional[Type[Any]] = None,
    file: TextIO = sys.stdout
) -> None:
    """Print the entries of the given matrix (or vector) in a easily readable format
    matrix      --  Matrix or vector to print
    entryFmt    -- A custom format string to use for formatting the matrix entries
    colsPerLine -- The maximum amount of columns displayed on a single line (will wrap, if needed)
    useLabels   -- Whether rows and columns should be labeled explicitly
    triangle    -- Only output the (lower) triangle of the matrix
    dtype       -- The type as which to print the matrix entries (only has effect if not entryFmt is provided)
    file        -- The file-like object to write to. Defaults to stdout
    """

    # Check matrix dimensions
    nRows = len(matrix)
    try:
        nColumns = len(matrix[0]) if nRows > 0 else 0
    except TypeError:
        # column vector -> transform to 1-column matrix to make it indexable by two indices
        nColumns = 1
        newMatrix = []
        for entry in matrix:
            newMatrix.append([entry])
        matrix = newMatrix

    if nRows == 0 and nColumns == 0:
        return

    # Determine the data type of the matrix entries
    if not dtype:
        try:
            # For numpy objects, we use the provided member
            dtype = matrix.dtype  # type: ignore
        except:
            # Otherwise, we assume that all entries have the same type and deduce the type
            # from the first entry (defaulting to float if there are no entries)
            dtype = type(matrix[0][0]) if nRows > 1 and nColumns > 1 else float

    # Choose appropriate format for the matrix entries
    if entryFmt is None:
        if np.issubdtype(dtype, int):
            # We use a float format to allow printing floats as integers
            entryFmt = "{: 5.0f}"
        elif np.issubdtype(dtype, complex):
            entryFmt = "{0.real: 7.2f} {0.imag:< 7.2f}"
        else:
            entryFmt = "{: 12.6f}"
    elif not "{" in entryFmt:
        entryFmt = "{:" + entryFmt + "}"

    colBatches = math.ceil(nColumns / colsPerLine)

    entryWidth = getFormatWidth(entryFmt)
    rowLabelWidth = math.ceil(math.log10(nRows) + 1e-6)

    colLabelFmt = "{:^" + str(entryWidth) + "d}"
    rowLabelFmt = "{:" + str(rowLabelWidth) + "d}"

    # Do the actual printing (in batches, in case the matrix has more columns that colsPerLine
    for batchNum in range(colBatches):
        # Correct for column batching (line wrapping)
        colOffset = batchNum * colsPerLine
        nColsInBatch = min(colsPerLine, nColumns - colOffset)

        # Handle printing only the lower triangle of the matrix
        nPrintRows = nRows - batchNum * colsPerLine if triangle else nRows
        assert nPrintRows > 0
        startRow = nRows - nPrintRows
        assert startRow >= 0

        if useLabels:
            # Print column labels
            print((" " * (rowLabelWidth + 1)) + "|", end="", file=file)
            for col in range(nColsInBatch):
                print(colLabelFmt.format(batchNum * colsPerLine + col + 1), end="|", file=file)
            print("", file=file)

        for row in range(startRow, startRow + nPrintRows):
            # Handle printing only the lower triangle of the matrix
            nPrintCols = (
                min(row + 1 - startRow, nColsInBatch) if triangle else nColsInBatch
            )

            if useLabels:
                # Start row with row label
                print(rowLabelFmt.format(row + 1), end="  ", file=file)

            for col in range(nPrintCols):
                print(entryFmt.format(matrix[row][col + colOffset]), end=" ", file=file)

            print("", file=file)
        print("", file=file)


def printMatC(
    mat,
    formatstr: Optional[str] = None,
    maxcol: int = 10,
    put_labels: bool = True,
    triangle: bool = False,
) -> None:
    """Print a matrix of complex values
    mat        --  a matrix (as exception also a vector is accepted)
    formatstr  -- format the real and imaginary part using this format (valid python format required)
    maxcol     -- max. number of columns in each output batch
    put_labels -- output the labels for rows and columns
    triangle   -- only output the (lower) triangle
    """
    printMat(
        matrix=mat,
        entryFmt="{:" + formatstr + "}" if formatstr else None,
        colsPerLine=maxcol,
        useLabels=put_labels,
        triangle=triangle,
        dtype=complex,
    )


def printMatF(
    mat,
    formatstr: Optional[str] = None,
    maxcol: int = 10,
    put_labels: bool = True,
    triangle: bool = False,
) -> None:
    """Print a matrix of float values
    mat        --  a matrix (as exception also a vector is accepted)
    formatstr  -- format the real and imaginary part using this format (valid python format required)
    maxcol     -- max. number of columns in each output batch
    put_labels -- output the labels for rows and columns
    triangle   -- only output the (lower) triangle
    """
    printMat(
        matrix=mat,
        entryFmt="{:" + formatstr + "}" if formatstr else None,
        colsPerLine=maxcol,
        useLabels=put_labels,
        triangle=triangle,
        dtype=float,
    )
