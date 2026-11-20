#!/usr/bin/env python3

import contextlib
import importlib.util
import io
import os
import shutil
import tempfile
import unittest

import pandas as pd

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, "data", "molpro_xml")

# Load the script as module (the scripts directory is not a package)
_spec = importlib.util.spec_from_file_location("extract_data", os.path.join(script_dir, "..", "scripts", "extract_data.py"))
assert _spec is not None and _spec.loader is not None
extract_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(extract_data)

MULTI_STEP = os.path.join(data_dir, "multi_step.xml")
DISPERSION = os.path.join(data_dir, "dft_dispersion.xml")


class TestExtractDataScript(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def run_script(self, *args):
        """Returns (exit code, stdout, stderr)"""
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                extract_data.main(list(args))
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else 1
                if isinstance(e.code, str):
                    err.write(e.code)
        return code, out.getvalue(), err.getvalue()

    def test_default_output_is_a_table(self):
        code, out, _ = self.run_script("-q", MULTI_STEP)
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines()[0].split()[:3], ["molecule", "basis_set", "functional"])
        self.assertIn("Exc. energy", out)

    def test_output_file_format_from_suffix(self):
        self.assertEqual(self.run_script("-q", MULTI_STEP, "-o", self.path("r.csv"))[0], 0)
        self.assertEqual(len(pd.read_csv(self.path("r.csv"), sep=";")), 4)

    def test_pivot(self):
        code, out, _ = self.run_script("-q", MULTI_STEP, DISPERSION, "--pivot", "command", "-p", "Energy")
        self.assertEqual(code, 0)
        self.assertIn("DF-RKS", out.splitlines()[0])

    def test_merge_creates_and_extends_storage(self):
        store = self.path("store.csv")
        self.assertEqual(self.run_script("-q", MULTI_STEP, "--merge-into", store)[0], 0)
        self.assertEqual(len(pd.read_csv(store, sep=";")), 4)
        code, _, err = self.run_script(DISPERSION, "--merge-into", store)
        self.assertEqual(code, 0)
        self.assertIn("4 existing and 1 new results, 5 in total", err)
        # Merging the same data again changes nothing
        self.run_script("-q", MULTI_STEP, DISPERSION, "--merge-into", store)
        frame = pd.read_csv(store, sep=";")
        self.assertEqual(len(frame), 5)
        self.assertEqual(sorted(frame["molecule"].unique()), ["ch", "water"])

    def test_merge_conflict_is_an_error_and_keeps_the_storage(self):
        shutil.copy(MULTI_STEP, self.path("a.xml"))
        with open(MULTI_STEP) as f:
            changed = f.read().replace("-38.5", "-38.6")
        with open(self.path("b.xml"), "w") as f:
            f.write(changed)
        store = self.path("store.csv")
        self.run_script("-q", self.path("a.xml"), "--merge-into", store)
        with open(store) as f:
            before = f.read()

        code, _, err = self.run_script("-q", self.path("b.xml"), "--merge-into", store)
        self.assertNotEqual(code, 0)
        self.assertIn("different values", err)
        with open(store) as f:
            self.assertEqual(f.read(), before)
        self.assertEqual([f for f in os.listdir(self.dir) if "tmp" in f], [])

        code, _, _ = self.run_script("-q", self.path("b.xml"), "--merge-into", store, "--on-conflict", "replace")
        self.assertEqual(code, 0)
        frame = pd.read_csv(store, sep=";")
        self.assertIn(-38.6, list(frame["value"]))
        self.assertNotIn(-38.5, list(frame["value"]))

    def test_labels_distinguish_conflicting_results(self):
        shutil.copy(MULTI_STEP, self.path("a.xml"))
        with open(MULTI_STEP) as f:
            changed = f.read().replace("-38.5", "-38.6")
        os.mkdir(self.path("variant"))
        with open(os.path.join(self.dir, "variant", "a.xml"), "w") as f:
            f.write(changed)
        store = self.path("store.csv")
        self.assertEqual(self.run_script("-q", self.path("a.xml"), "--merge-into", store)[0], 0)
        # Without a label of its own, the variant would conflict with the stored results
        self.assertNotEqual(self.run_script("-q", os.path.join(self.dir, "variant", "a.xml"), "--merge-into", store)[0], 0)
        code, _, _ = self.run_script("-q", os.path.join(self.dir, "variant", "a.xml"), "--label-from-path", "/(variant)/", "--merge-into", store)
        self.assertEqual(code, 0)
        frame = pd.read_csv(store, sep=";")
        self.assertEqual(list(frame["label"].dropna().unique()), ["variant"])
        self.assertEqual(int(frame["label"].isna().sum()), 4)
        self.assertEqual(len(frame), 8)

    def test_label_column_is_shown_only_if_used(self):
        self.assertNotIn("label", self.run_script("-q", MULTI_STEP)[1].splitlines()[0])
        self.assertIn("label", self.run_script("-q", MULTI_STEP, "--label-from-input", r"^basis=(\S+)")[1].splitlines()[0])

    def test_invalid_label_regex(self):
        code, _, err = self.run_script("-q", MULTI_STEP, "--label-from-path", "(")
        self.assertNotEqual(code, 0)
        self.assertIn("regular expression", err)

    def test_merge_into_json(self):
        store = self.path("store.json")
        self.run_script("-q", MULTI_STEP, "--merge-into", store)
        self.run_script("-q", DISPERSION, "--merge-into", store)
        self.assertEqual(len(pd.read_json(store)), 5)

    def test_invalid_merge_requests(self):
        for args in (
            ["--merge-into", self.path("s.csv"), "-o", self.path("o.csv")],
            ["--merge-into", self.path("s.csv"), "--pivot", "command"],
            ["--merge-into", self.path("s.txt")],
        ):
            code, _, err = self.run_script("-q", MULTI_STEP, *args)
            self.assertNotEqual(code, 0, args)
            self.assertNotEqual(err, "", args)
        self.assertEqual(os.listdir(self.dir), [])


if __name__ == "__main__":
    unittest.main()
