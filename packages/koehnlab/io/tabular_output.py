"""Writing pandas DataFrames in various file formats.

Formats are looked up in a registry by name or by file suffix. Additional formats can be added with :func:`register_format`.
Writers that need optional libraries (Parquet, Excel) report a helpful error if the library is missing.
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

DEFAULT_FORMAT = "table"
CSV_SEPARATOR = ";"


@dataclass
class OutputFormat:
    name: str
    suffixes: tuple
    # Called as write(frame, target) with target being a path or an open file object
    write: Callable
    # Binary formats need a binary stream if the target is not a path
    binary: bool = False
    # Called as read(path) and returning a DataFrame. None if the format cannot be read back (e.g. console tables)
    read: Optional[Callable] = None


_FORMATS: dict = {}


def register_format(fmt: OutputFormat) -> None:
    _FORMATS[fmt.name] = fmt


def available_formats() -> list:
    return list(_FORMATS)


def readable_formats() -> list:
    """Names of the formats that read_table can read"""
    return [name for name, fmt in _FORMATS.items() if fmt.read is not None]


def format_for_path(path: str) -> OutputFormat:
    """Determines the output format from the suffix of the path"""
    suffix = os.path.splitext(path)[1].lower()
    for fmt in _FORMATS.values():
        if suffix in fmt.suffixes:
            return fmt
    raise ValueError(
        "Cannot infer the output format from '%s'. Use one of the suffixes %s or specify the format explicitly (%s)"
        % (path, ", ".join(s for f in _FORMATS.values() for s in f.suffixes), ", ".join(_FORMATS))
    )


def read_table(path: str, fmt: Optional[str] = None) -> pd.DataFrame:
    """Reads a table written by write_table. The format is given by name or inferred from the suffix of the path.

    Note that formats like CSV do not store column types, so callers may have to convert the columns.
    """
    output_format = _FORMATS[fmt] if fmt is not None and fmt in _FORMATS else None
    if fmt is not None and output_format is None:
        raise ValueError("Unknown format '%s'. Available formats: %s" % (fmt, ", ".join(_FORMATS)))
    if output_format is None:
        output_format = format_for_path(path)
    if output_format.read is None:
        raise ValueError(
            "Tables in the format '%s' cannot be read back. Readable formats: %s"
            % (output_format.name, ", ".join(readable_formats()))
        )
    return output_format.read(path)


def write_table(frame: pd.DataFrame, destination: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """Writes the table to the destination.

    The destination is a file path or None for stdout. The format is given by name or, if omitted, inferred from the suffix of
    the destination. Writing to stdout defaults to a human-readable table.
    """
    if fmt is not None:
        if fmt not in _FORMATS:
            raise ValueError("Unknown output format '%s'. Available formats: %s" % (fmt, ", ".join(_FORMATS)))
        output_format = _FORMATS[fmt]
    elif destination is not None:
        output_format = format_for_path(destination)
    else:
        output_format = _FORMATS[DEFAULT_FORMAT]

    if destination is not None:
        output_format.write(frame, destination)
    else:
        output_format.write(frame, sys.stdout.buffer if output_format.binary else sys.stdout)


def _write_text(text: str, target) -> None:
    if isinstance(target, (str, os.PathLike)):
        with open(target, "w") as f:
            f.write(text)
    else:
        target.write(text)


def _write_table(frame: pd.DataFrame, target) -> None:
    # Going through object columns prints floats with full precision (the default would round energies to 6 decimals) and
    # shows missing values consistently for all column types
    shown = frame.astype(object).where(frame.notna(), "-")
    _write_text(shown.to_string(index=False) + "\n", target)


def _write_markdown(frame: pd.DataFrame, target) -> None:
    try:
        text = frame.to_markdown(index=False)
    except ImportError as e:
        raise ImportError("Writing Markdown requires the 'tabulate' package (pip3 install tabulate)") from e
    _write_text(text + "\n", target)


def _write_json(frame: pd.DataFrame, target) -> None:
    _write_text(frame.to_json(orient="records", indent=2, date_format="iso") + "\n", target)


def _write_jsonl(frame: pd.DataFrame, target) -> None:
    _write_text(frame.to_json(orient="records", lines=True, date_format="iso"), target)


def _write_excel(frame: pd.DataFrame, target) -> None:
    # Excel cannot store time zones
    frame = frame.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].dt.tz_convert("UTC").dt.tz_localize(None)
    frame.to_excel(target, index=False)


def _optional(function: Callable, library: str, pip_name: str) -> Callable:
    def write(frame, target):
        try:
            function(frame, target)
        except ImportError as e:
            raise ImportError("Writing this format requires the '%s' package (pip3 install %s)" % (library, pip_name)) from e

    return write


def _optional_read(function: Callable, library: str, pip_name: str) -> Callable:
    def read(path):
        try:
            return function(path)
        except ImportError as e:
            raise ImportError("Reading this format requires the '%s' package (pip3 install %s)" % (library, pip_name)) from e

    return read


for _fmt in (
    OutputFormat("table", (".txt",), _write_table),
    OutputFormat(
        "csv", (".csv",), lambda frame, target: frame.to_csv(target, sep=CSV_SEPARATOR, index=False),
        read=lambda path: pd.read_csv(path, sep=CSV_SEPARATOR),
    ),
    OutputFormat(
        "tsv", (".tsv",), lambda frame, target: frame.to_csv(target, sep="\t", index=False),
        read=lambda path: pd.read_csv(path, sep="\t"),
    ),
    OutputFormat("json", (".json",), _write_json, read=lambda path: pd.read_json(path, orient="records")),
    OutputFormat("jsonl", (".jsonl", ".ndjson"), _write_jsonl, read=lambda path: pd.read_json(path, orient="records", lines=True)),
    OutputFormat("markdown", (".md",), _write_markdown),
    OutputFormat(
        "parquet", (".parquet",), _optional(lambda f, t: f.to_parquet(t, index=False), "pyarrow", "pyarrow"), binary=True,
        read=_optional_read(pd.read_parquet, "pyarrow", "pyarrow"),
    ),
    OutputFormat(
        "excel", (".xlsx",), _optional(_write_excel, "openpyxl", "openpyxl"), binary=True,
        read=_optional_read(pd.read_excel, "openpyxl", "openpyxl"),
    ),
):
    register_format(_fmt)
