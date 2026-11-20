"""Collects results of Molpro XML files into a pandas DataFrame in tidy (long) format: one row per extracted property value.

Every aspect of a result (molecule, basis set, method, state, ...) is a column, so that new methods, basis sets or molecules
require no change of the structure. Use the usual pandas tools (``query``, ``pivot_table``, ``groupby``, ...) to reshape the
table, and see :mod:`koehnlab.io.tabular_output` for writing it to files.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .molpro_extract import ExtractionSpec, select_records
from .molpro_xml import Record, find_xml_files, name_molecules, parse_molpro_xml

COLUMNS = (
    "molecule", "formula", "label", "basis_set", "functional", "command", "property", "method", "multiplicity",
    "symmetry", "state", "value", "unit", "principal", "messages", "date", "source_path", "job_index", "step_index",
)

_CATEGORICAL = ("molecule", "formula", "label", "basis_set", "functional", "command", "property", "method", "unit")
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
            "molecule": r.molecule, "formula": r.formula, "label": r.label, "basis_set": r.basis_set, "functional": r.functional,
            "command": r.command, "property": r.property, "method": r.method, "multiplicity": r.multiplicity,
            "symmetry": r.symmetry, "state": r.state, "value": r.value, "unit": unit_of(r.property),
            "principal": r.principal, "messages": "; ".join(r.messages), "date": r.date, "source_path": r.source_path,
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
    frame["date"] = pd.to_datetime(frame["date"], utc=True, format="ISO8601")
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


# Columns that identify a result. Deliberately excludes the origin (source_path, job_index, step_index, date), so that
# several runs of the same calculation are recognized as the same result.
KEY_COLUMNS = (
    "molecule", "basis_set", "functional", "command", "property", "method", "multiplicity", "symmetry", "state",
)

CONFLICT_POLICIES = ("error", "newer", "keep-existing", "replace", "keep-both")


class MergeConflictError(ValueError):
    """Raised if results with the same key but different values are merged and the policy is 'error'"""

    def __init__(self, conflicts: pd.DataFrame):
        self.conflicts = conflicts
        n_keys = len(conflicts.groupby(list(KEY_COLUMNS), dropna=False, observed=True))
        lines = ["%d result(s) exist with different values (first conflicts shown):" % n_keys]
        for _, row in conflicts.head(6).iterrows():
            key = ", ".join("%s=%s" % (c, row[c]) for c in KEY_COLUMNS if not pd.isna(row[c]))
            lines.append("  %s: value %r from %s" % (key, row["value"], row["source_path"]))
        lines.append("Choose how to resolve conflicts with on_conflict / --on-conflict (%s)" % ", ".join(CONFLICT_POLICIES[1:]))
        super().__init__("\n".join(lines))


def conform_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Brings a table, e.g. one read from a CSV file, into the format created by to_dataframe.

    Missing columns are added (as missing values) and the columns of COLUMNS are put first. Additional columns are kept.
    """
    frame = frame.copy()
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[list(COLUMNS) + [c for c in frame.columns if c not in COLUMNS]]
    for column in _NULLABLE_INT:
        frame[column] = pd.to_numeric(frame[column]).astype("Int64")
    for column in _CATEGORICAL:
        frame[column] = frame[column].astype("category")
    frame["value"] = pd.to_numeric(frame["value"]).astype("float64")
    frame["principal"] = frame["principal"].fillna(False).astype("bool")
    frame["messages"] = frame["messages"].fillna("").astype(str)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, format="ISO8601")
    return frame


def merge_tables(existing: pd.DataFrame, new: pd.DataFrame, on_conflict: str = "error") -> pd.DataFrame:
    """Merges the new results into an existing table.

    Results are identified by KEY_COLUMNS. A key that occurs several times (in either table or in both) with identical values is
    reduced to a single row (the first), which makes merging the same data repeatedly harmless. If the values differ, on_conflict
    decides:

    - "error": raise MergeConflictError, nothing is merged
    - "newer": keep the result with the latest date (results without date count as oldest; the later row wins ties)
    - "keep-existing": keep the first result of the existing table
    - "replace": keep the last new result
    - "keep-both": keep all of them

    Returns a new table, the arguments are not modified. Rows keep their order: existing ones first, then the new ones.
    """
    if on_conflict not in CONFLICT_POLICIES:
        raise ValueError("Unknown conflict policy '%s'. Choose from: %s" % (on_conflict, ", ".join(CONFLICT_POLICIES)))

    existing = conform_table(existing)
    new = conform_table(new)
    combined = pd.concat([existing.astype(object), new.astype(object)], ignore_index=True)
    is_new = np.arange(len(combined)) >= len(existing)

    group = combined[list(KEY_COLUMNS)].groupby(list(KEY_COLUMNS), dropna=False, sort=False).ngroup()
    keep = np.ones(len(combined), dtype=bool)
    dates = pd.to_datetime(combined["date"], utc=True)
    conflicting = []

    duplicated = group[group.map(group.value_counts()) > 1]
    for rows in duplicated.groupby(duplicated).groups.values():
        rows = list(rows)
        values = combined.loc[rows, "value"].astype(float)
        if values.nunique(dropna=False) == 1:
            keep[rows[1:]] = False
        elif on_conflict == "error":
            conflicting.extend(rows)
        elif on_conflict == "newer":
            winner = max(rows, key=lambda r: (dates[r] if not pd.isna(dates[r]) else pd.Timestamp.min.tz_localize("UTC"), r))
            keep[[r for r in rows if r != winner]] = False
        elif on_conflict == "keep-existing":
            existing_rows = [r for r in rows if not is_new[r]]
            winner = existing_rows[0] if existing_rows else rows[-1]
            keep[[r for r in rows if r != winner]] = False
        elif on_conflict == "replace":
            new_rows = [r for r in rows if is_new[r]]
            winner = new_rows[-1] if new_rows else rows[-1]
            keep[[r for r in rows if r != winner]] = False
        # "keep-both" keeps everything

    if conflicting:
        raise MergeConflictError(conform_table(combined.loc[sorted(conflicting)]))
    return conform_table(combined[keep].reset_index(drop=True))
