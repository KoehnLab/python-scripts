"""Collects results of Molpro XML files into a pandas DataFrame in tidy (long) format: one row per extracted property value.

Every aspect of a result (molecule, basis set, method, state, ...) is a column, so that new methods, basis sets or molecules
require no change of the structure. Use the usual pandas tools (``query``, ``pivot_table``, ``groupby``, ...) to reshape the
table, and see :mod:`koehnlab.io.tabular_output` for writing it to files.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

from .molpro_extract import ExtractionSpec, select_records
from .molpro_xml import Record, find_xml_files, name_molecules, parse_molpro_xml

COLUMNS = (
    "molecule", "formula", "basis_set", "functional", "command", "property", "method", "multiplicity", "symmetry",
    "state", "value", "unit", "principal", "messages", "source_path", "job_index", "step_index",
)

_CATEGORICAL = ("molecule", "formula", "basis_set", "functional", "command", "property", "method", "unit")
_NULLABLE_INT = ("multiplicity", "symmetry", "state", "job_index", "step_index")


def unit_of(property_name: str) -> Optional[str]:
    """Unit in which Molpro reports the given property (atomic units), if known"""
    name = property_name.lower()
    if "energy" in name or "correction" in name:
        return "E_h"
    if "dipole" in name:
        return "a.u."
    return None


def to_dataframe(records: Iterable[Record]) -> pd.DataFrame:
    """Converts the records into a DataFrame with the columns given by COLUMNS and well-defined dtypes"""
    rows = [
        {
            "molecule": r.molecule, "formula": r.formula, "basis_set": r.basis_set, "functional": r.functional,
            "command": r.command, "property": r.property, "method": r.method, "multiplicity": r.multiplicity,
            "symmetry": r.symmetry, "state": r.state, "value": r.value, "unit": unit_of(r.property),
            "principal": r.principal, "messages": "; ".join(r.messages), "source_path": r.source_path,
            "job_index": r.job_index, "step_index": r.step_index,
        }
        for r in records
    ]
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    for column in _NULLABLE_INT:
        frame[column] = frame[column].astype("Int64")
    for column in _CATEGORICAL:
        frame[column] = frame[column].astype("category")
    frame["value"] = frame["value"].astype("float64")
    frame["principal"] = frame["principal"].astype("bool")
    return frame


@dataclass
class Extraction:
    table: pd.DataFrame
    # (path, description) of everything that went wrong or deserves attention
    problems: list = field(default_factory=list)


def extract_table(paths: Iterable[str], spec: ExtractionSpec = ExtractionSpec()) -> Extraction:
    """Parses all XML files found at the given paths (files or directories to search recursively) into one table"""
    records: list = []
    problems: list = []
    for path in paths:
        for xml_file in find_xml_files(path):
            result = parse_molpro_xml(xml_file)
            name_molecules(result)
            records.extend(select_records(result, spec))
            problems.extend((xml_file, problem) for problem in result.problems)
    return Extraction(table=to_dataframe(records), problems=problems)
