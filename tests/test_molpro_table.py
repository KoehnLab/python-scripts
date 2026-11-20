#!/usr/bin/env python3

import os
import unittest

import pandas as pd

from koehnlab.io.molpro_extract import ExtractionSpec
from koehnlab.io.molpro_table import COLUMNS, extract_table, to_dataframe, unit_of

script_dir: str = os.path.dirname(os.path.realpath(__file__))
data_dir: str = os.path.join(script_dir, "data", "molpro_xml")


class TestMolproTable(unittest.TestCase):
    def setUp(self):
        self.extraction = extract_table([data_dir])
        self.table = self.extraction.table

    def test_columns_and_dtypes(self):
        self.assertEqual(list(self.table.columns), list(COLUMNS))
        self.assertEqual(self.table["value"].dtype, "float64")
        self.assertEqual(self.table["state"].dtype, "Int64")
        self.assertEqual(self.table["molecule"].dtype, "category")
        self.assertEqual(self.table["principal"].dtype, "bool")

    def test_content(self):
        # 4 rows from multi_step.xml and 1 from dft_dispersion.xml; invalid.xml does not contribute
        self.assertEqual(len(self.table), 5)
        cmrcc = self.table[self.table["command"] == "DF-CMRCC"]
        self.assertEqual(list(cmrcc["property"]), ["Total energy", "Exc. energy"])
        self.assertEqual(list(cmrcc["symmetry"]), [1, 2])
        self.assertEqual(set(cmrcc["basis_set"]), {"cc-pVDZ"})
        self.assertEqual(set(cmrcc["unit"]), {"E_h"})
        self.assertIn("Reference energies differ", cmrcc["messages"].iloc[0])

    def test_molecules_and_basis_sets_share_one_table(self):
        self.assertEqual(set(self.table["molecule"]), {"ch", "water"})
        self.assertEqual(set(self.table["basis_set"]), {"cc-pVDZ", "def2-SVP", "DEF2-TZVP"})
        by_molecule = self.table.pivot_table(index="molecule", columns="basis_set", values="value", observed=True)
        self.assertEqual(by_molecule.shape, (2, 3))

    def test_date_is_utc_datetime(self):
        self.assertEqual(str(self.table["date"].dtype), "datetime64[ns, UTC]".replace("ns", self.table["date"].dtype.unit))
        dated = self.table[self.table["molecule"] == "ch"]["date"]
        self.assertEqual(set(dated), {pd.Timestamp("2026-11-19T13:07:21", tz="UTC")})
        self.assertTrue(self.table[self.table["molecule"] == "water"]["date"].isna().all())

    def test_label_column(self):
        self.assertTrue(self.table["label"].isna().all())
        table = extract_table([data_dir], ExtractionSpec(label="CAS(1,1)")).table
        self.assertEqual(set(table["label"]), {"CAS(1,1)"})

    def test_problems_are_collected(self):
        problems = {(os.path.basename(path), text) for path, text in self.extraction.problems}
        self.assertTrue(any(name == "invalid.xml" and text.startswith("Invalid XML") for name, text in problems))
        self.assertTrue(any(name == "multi_step.xml" and "Reference energies differ" in text for name, text in problems))

    def test_spec_is_applied(self):
        table = extract_table([data_dir], ExtractionSpec(properties=("Exc. energy",))).table
        self.assertEqual(list(table["value"]), [0.25])

    def test_empty_table_has_all_columns(self):
        table = to_dataframe([])
        self.assertEqual(list(table.columns), list(COLUMNS))
        self.assertEqual(len(table), 0)

    def test_units(self):
        self.assertEqual(unit_of("Exc. energy"), "E_h")
        self.assertEqual(unit_of("Dipole moment"), "a.u.")
        self.assertIsNone(unit_of("Virial quotient"))


if __name__ == "__main__":
    unittest.main()
