#!/usr/bin/env python3

import unittest

import pandas as pd

from koehnlab.io.molpro_table import CONFLICT_POLICIES, KEY_COLUMNS, MergeConflictError, conform_table, merge_tables, to_dataframe
from koehnlab.io.molpro_xml import Record


def record(value, state=1, molecule="a", basis_set="vdz", source="run1.xml", date="2026-01-01T00:00:00+00:00", **kwargs):
    fields = dict(
        source_path=source, job_index=0, step_index=0, command="CMRCC", commandset=None, parent_command=None,
        last_nested=False, property="Exc. energy", method="LR-MRCC2", value=value, raw_value=str(value), principal=True,
        state=state, symmetry=1, multiplicity=2, basis_set=basis_set, functional=None, molecule=molecule, date=date,
    )
    fields.update(kwargs)
    return Record(**fields)


def table(*records):
    return to_dataframe(records)


class TestMerge(unittest.TestCase):
    def values(self, frame):
        return list(frame["value"])

    def test_disjoint_tables_are_concatenated(self):
        merged = merge_tables(table(record(1.0, state=1)), table(record(2.0, state=2), record(3.0, molecule="b")))
        self.assertEqual(self.values(merged), [1.0, 2.0, 3.0])
        self.assertEqual(merged["state"].dtype, "Int64")
        self.assertEqual(merged["molecule"].dtype, "category")

    def test_identical_results_are_merged_once(self):
        existing = table(record(1.0), record(2.0, state=2))
        merged = merge_tables(existing, existing)
        self.assertEqual(self.values(merged), [1.0, 2.0])
        # Merging is idempotent also if the origin differs but the value is the same
        merged = merge_tables(existing, table(record(1.0, source="rerun.xml", date="2026-02-01T00:00:00+00:00")))
        self.assertEqual(self.values(merged), [1.0, 2.0])
        self.assertEqual(merged["source_path"].iloc[0], "run1.xml")

    def test_conflict_raises_by_default(self):
        with self.assertRaises(MergeConflictError) as ctx:
            merge_tables(table(record(1.0)), table(record(1.5, source="run2.xml")))
        self.assertIn("run2.xml", str(ctx.exception))
        self.assertEqual(len(ctx.exception.conflicts), 2)

    def test_conflict_within_new_results(self):
        with self.assertRaises(MergeConflictError):
            merge_tables(table(), table(record(1.0), record(2.0, source="run2.xml")))

    def test_newer(self):
        old = record(1.0, date="2026-01-01T00:00:00+00:00")
        new = record(2.0, source="run2.xml", date="2026-03-01T00:00:00+00:00")
        self.assertEqual(self.values(merge_tables(table(old), table(new), "newer")), [2.0])
        # Also if the newer result is in the existing table
        self.assertEqual(self.values(merge_tables(table(new), table(old), "newer")), [2.0])

    def test_newer_without_dates(self):
        undated = record(1.0, date=None)
        dated = record(2.0, date="2026-01-01T00:00:00+00:00", source="run2.xml")
        self.assertEqual(self.values(merge_tables(table(dated), table(undated), "newer")), [2.0])
        # Equal dates: the later row wins
        a, b = record(1.0, date=None), record(2.0, date=None, source="run2.xml")
        self.assertEqual(self.values(merge_tables(table(a), table(b), "newer")), [2.0])

    def test_keep_existing_and_replace(self):
        existing, new = table(record(1.0)), table(record(2.0, source="run2.xml"), record(3.0, source="run3.xml"))
        self.assertEqual(self.values(merge_tables(existing, new, "keep-existing")), [1.0])
        self.assertEqual(self.values(merge_tables(existing, new, "replace")), [3.0])

    def test_keep_both(self):
        merged = merge_tables(table(record(1.0)), table(record(2.0, source="run2.xml")), "keep-both")
        self.assertEqual(self.values(merged), [1.0, 2.0])

    def test_conflicts_only_affect_their_key(self):
        existing = table(record(1.0, state=1), record(5.0, state=2))
        new = table(record(1.5, state=1, source="run2.xml"), record(7.0, state=3))
        merged = merge_tables(existing, new, "replace")
        self.assertEqual(self.values(merged), [5.0, 1.5, 7.0])

    def test_missing_values_in_key_columns_match(self):
        # functional is missing for wave function methods
        merged = merge_tables(table(record(1.0)), table(record(1.0, source="r2.xml")))
        self.assertEqual(len(merged), 1)

    def test_source_does_not_matter_but_basis_set_does(self):
        merged = merge_tables(table(record(1.0)), table(record(2.0, basis_set="vtz")))
        self.assertEqual(len(merged), 2)

    def test_empty_tables(self):
        self.assertEqual(len(merge_tables(table(), table())), 0)
        self.assertEqual(self.values(merge_tables(table(), table(record(1.0)))), [1.0])
        self.assertEqual(self.values(merge_tables(table(record(1.0)), table())), [1.0])

    def test_unknown_policy(self):
        with self.assertRaises(ValueError):
            merge_tables(table(), table(), "nope")

    def test_arguments_are_not_modified(self):
        existing, new = table(record(1.0)), table(record(2.0, state=2))
        merge_tables(existing, new)
        self.assertEqual(len(existing), 1)
        self.assertEqual(len(new), 1)

    def test_policies_are_listed(self):
        self.assertEqual(CONFLICT_POLICIES[0], "error")
        self.assertNotIn("source_path", KEY_COLUMNS)


class TestConform(unittest.TestCase):
    def test_csv_like_frame(self):
        frame = pd.DataFrame(
            {"molecule": ["a"], "state": [1.0], "value": ["1.5"], "principal": [True], "messages": [None], "date": ["2026-01-01 00:00:00+00:00"]}
        )
        conformed = conform_table(frame)
        self.assertEqual(conformed["state"].dtype, "Int64")
        self.assertEqual(conformed["value"].dtype, "float64")
        self.assertEqual(conformed["messages"].iloc[0], "")
        self.assertEqual(conformed["date"].iloc[0], pd.Timestamp("2026-01-01", tz="UTC"))
        self.assertEqual(list(conformed.columns)[:2], ["molecule", "formula"])

    def test_additional_columns_are_kept(self):
        frame = table(record(1.0)).assign(comment="x")
        self.assertEqual(list(conform_table(frame).columns)[-1], "comment")


if __name__ == "__main__":
    unittest.main()
