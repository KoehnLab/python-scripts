#!/usr/bin/env python3

import os
import unittest

from koehnlab.io.molpro_xml import parse_molpro_xml, find_xml_files

script_dir: str = os.path.dirname(os.path.realpath(__file__))
data_dir: str = os.path.join(script_dir, "data", "molpro_xml")


class TestMolproXmlParser(unittest.TestCase):
    def setUp(self):
        self.result = parse_molpro_xml(os.path.join(data_dir, "multi_step.xml"))

    def test_all_properties_are_read(self):
        self.assertEqual(len(self.result.records), 7)

    def test_property_attributes(self):
        energy = self.result.records[0]
        self.assertEqual(energy.property, "Energy")
        self.assertEqual(energy.method, "RHF")
        self.assertEqual(energy.value, -38.0)
        self.assertTrue(energy.principal)
        self.assertEqual((energy.state, energy.symmetry, energy.multiplicity), (1, 1, 2))
        self.assertEqual(energy.command, "DF-RHF-SCF")
        self.assertEqual(energy.elements, ("C", "H"))

    def test_non_principal_and_non_scalar(self):
        kinetic = self.result.records[1]
        self.assertFalse(kinetic.principal)
        self.assertIsNone(kinetic.state)
        dipole = self.result.records[2]
        self.assertIsNone(dipole.value)
        self.assertEqual(dipole.raw_value, "0.0 0.0 0.115880219390735E-01")

    def test_context_is_inherited_by_steps_without_summary(self):
        # The summary comes after the properties and the CMRCC step has none at all
        cmrcc = [r for r in self.result.records if r.command == "DF-CMRCC"]
        self.assertEqual(len(cmrcc), 2)
        for r in cmrcc:
            self.assertEqual(r.basis_set, "cc-pVDZ")
            self.assertEqual(r.elements, ("C", "H"))
            self.assertIsNone(r.functional)
        self.assertEqual(self.result.records[0].basis_set, "cc-pVDZ")

    def test_messages(self):
        cmrcc = [r for r in self.result.records if r.command == "DF-CMRCC"]
        self.assertEqual(cmrcc[0].messages, ("Warning: Reference energies differ",))
        self.assertFalse(cmrcc[0].errored)
        self.assertTrue(any("Reference energies differ" in p for p in self.result.problems))

    def test_nested_steps(self):
        nested = [r for r in self.result.records if r.parent_command == "OPTG"]
        self.assertEqual([r.value for r in nested], [-1.0, -1.5])
        self.assertEqual([r.last_nested for r in nested], [False, True])
        self.assertEqual({r.functional for r in nested}, {"B3LYP"})
        self.assertEqual({r.basis_set for r in nested}, {"def2-SVP"})
        # step_index counts nested steps as well
        self.assertEqual([r.step_index for r in nested], [3, 4])

    def test_input_is_available(self):
        text = self.result.inputs[0]
        self.assertIn("geometry=/some/path/ch.xyz", text)
        self.assertIn("include /some/path/ch.xyz", text)

    def test_invalid_xml_is_reported(self):
        result = parse_molpro_xml(os.path.join(data_dir, "invalid.xml"))
        self.assertEqual(result.records, [])
        self.assertEqual(len(result.problems), 1)

    def test_missing_file_is_reported(self):
        result = parse_molpro_xml(os.path.join(data_dir, "does_not_exist.xml"))
        self.assertEqual(len(result.problems), 1)

    def test_find_xml_files(self):
        found = find_xml_files(data_dir)
        self.assertEqual([os.path.basename(f) for f in found], ["invalid.xml", "multi_step.xml"])
        self.assertEqual(find_xml_files(found[1]), [found[1]])


if __name__ == "__main__":
    unittest.main()
