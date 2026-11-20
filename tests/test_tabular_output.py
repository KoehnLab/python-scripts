#!/usr/bin/env python3

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

import pandas as pd

from koehnlab.io.tabular_output import available_formats, format_for_path, read_table, write_table


def example_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "molecule": pd.Series(["a", "b"], dtype="category"),
            "state": pd.Series([1, None], dtype="Int64"),
            "value": [-1.5, 0.25],
        }
    )


class TestTabularOutput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def path(self, name: str) -> str:
        return os.path.join(self.tmp.name, name)

    def test_default_to_stdout_is_a_table(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            write_table(example_table())
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0].split(), ["molecule", "state", "value"])
        self.assertEqual(lines[2].split(), ["b", "-", "0.25"])

    def test_csv_roundtrip(self):
        write_table(example_table(), self.path("t.csv"))
        with open(self.path("t.csv")) as f:
            self.assertEqual(f.readline().strip(), "molecule;state;value")
        frame = pd.read_csv(self.path("t.csv"), sep=";")
        self.assertEqual(list(frame["value"]), [-1.5, 0.25])
        self.assertTrue(frame["state"].isna().iloc[1])

    def test_tsv_and_explicit_format(self):
        write_table(example_table(), self.path("t.dat"), fmt="tsv")
        self.assertEqual(len(pd.read_csv(self.path("t.dat"), sep="\t")), 2)

    def test_json_and_jsonl_roundtrip(self):
        write_table(example_table(), self.path("t.json"))
        write_table(example_table(), self.path("t.jsonl"))
        for name in ("t.json", "t.jsonl"):
            frame = pd.read_json(self.path(name), lines=name.endswith(".jsonl"))
            self.assertEqual(list(frame["molecule"]), ["a", "b"])
            self.assertEqual(list(frame["value"]), [-1.5, 0.25])
            self.assertTrue(frame["state"].isna().iloc[1])

    def test_explicit_format_to_stdout(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            write_table(example_table(), fmt="csv")
        self.assertTrue(out.getvalue().startswith("molecule;state;value"))

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is not installed")
    def test_parquet_roundtrip(self):
        write_table(example_table(), self.path("t.parquet"))
        frame = pd.read_parquet(self.path("t.parquet"))
        pd.testing.assert_frame_equal(frame, example_table(), check_dtype=False, check_categorical=False)

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "openpyxl is not installed")
    def test_excel_roundtrip(self):
        write_table(example_table(), self.path("t.xlsx"))
        self.assertEqual(len(pd.read_excel(self.path("t.xlsx"))), 2)

    def test_unknown_format_and_suffix(self):
        with self.assertRaises(ValueError):
            write_table(example_table(), self.path("t.csv"), fmt="nope")
        with self.assertRaises(ValueError):
            format_for_path("t.unknown")

    def test_read_table_roundtrip(self):
        for name in ("t.csv", "t.tsv", "t.json", "t.jsonl"):
            write_table(example_table(), self.path(name))
            frame = read_table(self.path(name))
            self.assertEqual(list(frame.columns), ["molecule", "state", "value"], name)
            self.assertEqual(list(frame["value"]), [-1.5, 0.25], name)
            self.assertTrue(frame["state"].isna().iloc[1], name)

    def test_read_table_with_explicit_format(self):
        write_table(example_table(), self.path("t.dat"), fmt="csv")
        self.assertEqual(len(read_table(self.path("t.dat"), fmt="csv")), 2)

    def test_unreadable_and_unknown_formats(self):
        write_table(example_table(), self.path("t.txt"))
        with self.assertRaises(ValueError):
            read_table(self.path("t.txt"))
        with self.assertRaises(ValueError):
            read_table(self.path("t.csv"), fmt="nope")

    def test_registry(self):
        self.assertIn("csv", available_formats())
        self.assertEqual(format_for_path("X.CSV").name, "csv")


if __name__ == "__main__":
    unittest.main()
