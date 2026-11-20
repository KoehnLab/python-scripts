#!/usr/bin/env python3

import argparse
import os
import sys

from koehnlab.io.molpro_extract import ExtractionSpec, NESTED_STEP_MODES
from koehnlab.io.molpro_table import CONFLICT_POLICIES, conform_table, extract_table, merge_tables
from koehnlab.io.tabular_output import available_formats, format_for_path, read_table, readable_formats, write_table

DEFAULT_COLUMNS = [
    "molecule", "basis_set", "functional", "command", "property", "method", "multiplicity", "symmetry", "state",
    "value", "unit", "source_path",
]
PIVOT_INDEX = ["molecule", "functional", "command", "property", "method", "multiplicity", "symmetry", "state"]


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Collects the results of Molpro calculations from their XML output files into a table. "
        "Every row is a single result (e.g. an energy) together with the molecule, basis set, method and state it belongs to."
    )
    parser.add_argument("search_paths", nargs="+", help="XML files or directories that are searched recursively for XML files")
    parser.add_argument(
        "-o", "--output",
        help="Write the table to this file. The format is inferred from the suffix (%s unless --format is given). "
        "Without this option, the table is printed to stdout." % ", ".join(available_formats()),
    )
    parser.add_argument(
        "-f", "--format", choices=available_formats(),
        help="Output format. Defaults to the suffix of --output or to a human-readable table when printing to stdout.",
    )
    parser.add_argument(
        "-p", "--properties", nargs="+", metavar="NAME",
        help="Names of the properties to extract (case-insensitive, default: %s). Use quotes for names with spaces."
        % ", ".join('"%s"' % p for p in ExtractionSpec.properties),
    )
    parser.add_argument("--all-properties", action="store_true", help="Extract all properties instead of a selection")
    parser.add_argument("--include-non-principal", action="store_true", help="Also extract properties not flagged as principal")
    parser.add_argument(
        "--nested-steps", choices=NESTED_STEP_MODES, default=ExtractionSpec.nested_steps,
        help="Which steps nested in other steps (e.g. single points of a geometry optimization) to consider (default: %(default)s)",
    )
    parser.add_argument("--keep-errored", action="store_true", help="Do not skip job steps for which Molpro reported an error")
    parser.add_argument("--columns", nargs="+", metavar="COLUMN", help="Columns to output (default: %s)" % ", ".join(DEFAULT_COLUMNS))
    parser.add_argument("--all-columns", action="store_true", help="Output all available columns")
    parser.add_argument(
        "--pivot", metavar="COLUMN",
        help="Reshape the table such that the values of the given column (e.g. basis_set) become separate columns",
    )
    parser.add_argument(
        "-m", "--merge-into", metavar="FILE",
        help="Merge the results into this table (created if it does not exist) and write it back. The format is inferred from "
        "the suffix unless --format is given and has to be one that can be read back (not 'table' or 'markdown'). Results are "
        "identified by molecule, basis set, functional, command, property, method, multiplicity, symmetry and state. "
        "Cannot be combined with -o, --pivot or column selection.",
    )
    parser.add_argument(
        "--on-conflict", choices=CONFLICT_POLICIES, default="error",
        help="What to do if --merge-into finds results with the same identity but different values (e.g. the same calculation "
        "was run twice): 'error' (default) aborts without changing anything, 'newer' keeps the result with the latest date, "
        "'keep-existing' keeps the one already in FILE, 'replace' takes the new one, 'keep-both' keeps all of them",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Do not print warnings and problems to stderr")
    return parser.parse_args(argv)


def merge_into_file(table, args):
    """Merges the table into the storage file given by --merge-into and replaces that file"""
    if args.output or args.pivot or args.columns or args.all_columns:
        sys.exit("--merge-into writes the complete table to its file and cannot be combined with -o, --pivot or column options")

    path = args.merge_into
    fmt = args.format
    try:
        if fmt is None:
            fmt = format_for_path(path).name
        if fmt not in readable_formats():
            raise ValueError(
                "Cannot merge into a file of format '%s' as it cannot be read back. Readable formats: %s"
                % (fmt, ", ".join(readable_formats()))
            )
        existing = conform_table(read_table(path, fmt)) if os.path.exists(path) else table.iloc[0:0]
        merged = merge_tables(existing, table, args.on_conflict)
    except (ValueError, ImportError) as e:
        sys.exit(str(e))

    # Write to a temporary file first so that a failure cannot destroy the existing storage
    temporary = "%s.tmp-%d" % (path, os.getpid())
    try:
        write_table(merged, temporary, fmt)
        os.replace(temporary, path)
    except (ValueError, ImportError, OSError) as e:
        if os.path.exists(temporary):
            os.remove(temporary)
        sys.exit(str(e))

    if not args.quiet:
        print(
            "%s: %d existing and %d new results, %d in total" % (path, len(existing), len(table), len(merged)), file=sys.stderr
        )


def main(argv=None):
    args = parse_arguments(argv)

    spec = ExtractionSpec(
        properties=None if args.all_properties else (tuple(args.properties) if args.properties else ExtractionSpec.properties),
        principal_only=not args.include_non_principal,
        nested_steps=args.nested_steps,
        skip_errored_steps=not args.keep_errored,
    )
    extraction = extract_table(args.search_paths, spec)
    table = extraction.table

    if not args.quiet:
        for path, problem in extraction.problems:
            print("%s: %s" % (path, problem), file=sys.stderr)

    if args.merge_into:
        merge_into_file(table, args)
        return

    if args.pivot:
        if args.pivot not in table.columns:
            sys.exit("Cannot pivot on unknown column '%s'. Available columns: %s" % (args.pivot, ", ".join(table.columns)))
        index = [c for c in PIVOT_INDEX if c != args.pivot]
        # pivot_table drops rows with missing index values (e.g. no functional for wave function methods)
        table = table.astype({c: object for c in index}).fillna({c: "-" for c in index})
        duplicates = int(table.duplicated(subset=index + [args.pivot]).sum())
        if duplicates > 0 and not args.quiet:
            print(
                "Warning: %d results share all of %s; only the first one is shown (e.g. the same calculation was run "
                "several times)" % (duplicates, ", ".join(index + [args.pivot])),
                file=sys.stderr,
            )
        table = table.pivot_table(index=index, columns=args.pivot, values="value", aggfunc="first", observed=True).reset_index()
        table.columns.name = None
    else:
        columns = list(table.columns) if args.all_columns else (args.columns or DEFAULT_COLUMNS)
        unknown = [c for c in columns if c not in table.columns]
        if unknown:
            sys.exit("Unknown column(s): %s. Available columns: %s" % (", ".join(unknown), ", ".join(table.columns)))
        table = table[columns]

    try:
        write_table(table, args.output, args.format)
    except (ValueError, ImportError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
