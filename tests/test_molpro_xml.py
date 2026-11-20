#!/usr/bin/env python3

import os
import tempfile
import unittest

from koehnlab.io.molpro_extract import ExtractionSpec, select_records
from koehnlab.io.molpro_xml import (
    parse_molpro_xml, find_xml_files, find_input_file, hill_formula, xyz_names_from_input, name_from_path, name_molecules, ParseResult, Record
)

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

    def test_basis_from_overall_method(self):
        result = parse_molpro_xml(os.path.join(data_dir, "dft_dispersion.xml"))
        self.assertEqual({r.basis_set for r in result.records}, {"DEF2-TZVP"})

    def test_date(self):
        self.assertEqual({r.date for r in self.result.records}, {"2026-11-19T14:07:21+01:00"})
        result = parse_molpro_xml(os.path.join(data_dir, "dft_dispersion.xml"))
        self.assertEqual({r.date for r in result.records}, {None})

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
        self.assertEqual([os.path.basename(f) for f in found], ["dft_dispersion.xml", "invalid.xml", "multi_step.xml"])
        self.assertEqual(find_xml_files(found[2]), [found[2]])


def _record(job_index=0, elements=()):
    return Record(
        source_path="x", job_index=job_index, step_index=0, command="C", commandset=None, parent_command=None,
        last_nested=False, property="Energy", method=None, value=1.0, raw_value="1.0", principal=True, state=None,
        symmetry=None, multiplicity=None, basis_set=None, functional=None, elements=elements,
    )


class TestMoleculeNaming(unittest.TestCase):
    def test_hill_formula(self):
        self.assertEqual(hill_formula(["H", "C", "C", "C", "H", "H", "H", "H", "H"]), "C3H6")
        self.assertEqual(hill_formula(["O", "H", "H"]), "H2O")
        self.assertEqual(hill_formula(["N", "O", "O", "C", "H", "H", "H"]), "CH3NO2")
        self.assertEqual(hill_formula(["Cl", "Na"]), "ClNa")
        self.assertEqual(hill_formula(["Zn"]), "Zn")

    def test_xyz_names(self):
        self.assertEqual(xyz_names_from_input("basis=vdz\ngeometry=/a/b/allyl.xyz\nhf"), ["allyl"])
        self.assertEqual(xyz_names_from_input("geom=a.xyz\nhf\ngeometry=b.xyz\ngeometry=a.xyz"), ["a", "b"])
        # Inline geometries and other formats do not give a name
        self.assertEqual(xyz_names_from_input("geometry={O;H,O,1.0}\ngeometry=foo.zmat"), [])

    def test_name_from_path(self):
        self.assertEqual(name_from_path("/x/allyl_dz_gecco.20261119-130642-45592bf8557d.xml"), "allyl_dz_gecco")
        self.assertEqual(name_from_path("allyl.316816-20261020.xml"), "allyl")
        self.assertEqual(name_from_path("plain.xml"), "plain")

    def test_xyz_name_is_preferred(self):
        result = parse_molpro_xml(os.path.join(data_dir, "multi_step.xml"))
        name_molecules(result)
        self.assertEqual({r.molecule for r in result.records}, {"ch"})
        self.assertEqual(result.records[0].formula, "CH")

    def test_formula_fallback_for_inline_geometry(self):
        result = ParseResult(path="a.1-2.xml", inputs={0: "geometry={O;H,O,1.0}"}, records=[_record(elements=("O", "H", "H"))])
        name_molecules(result)
        self.assertEqual(result.records[0].molecule, "H2O")

    def test_multiple_molecules_in_one_job(self):
        records = [_record(elements=("O", "H", "H")), _record(elements=("O", "H", "H")), _record(elements=("C", "O", "O"))]
        result = ParseResult(path="a.xml", inputs={0: "geometry=water.xyz\ngeometry=co2.xyz"}, records=records)
        name_molecules(result)
        self.assertEqual([r.molecule for r in records], ["water", "water", "co2"])
        self.assertEqual([r.formula for r in records], ["H2O", "H2O", "CO2"])

    def test_ambiguous_xyz_names_fall_back_to_formula(self):
        records = [_record(elements=("O", "H", "H"))]
        result = ParseResult(path="a.xml", inputs={0: "geometry=a.xyz\ngeometry=b.xyz"}, records=records)
        name_molecules(result)
        self.assertEqual(records[0].molecule, "H2O")

    def test_file_name_as_last_resort(self):
        records = [_record()]
        result = ParseResult(path="/x/foo.20261119-130642-ab12.xml", records=records)
        name_molecules(result)
        self.assertEqual(records[0].molecule, "foo")
        self.assertIsNone(records[0].formula)

    def test_jobs_are_named_independently(self):
        records = [_record(job_index=0, elements=("O", "H", "H")), _record(job_index=1, elements=("C", "O", "O"))]
        result = ParseResult(path="a.xml", inputs={0: "geometry=water.xyz", 1: "geometry=co2.xyz"}, records=records)
        name_molecules(result)
        self.assertEqual([r.molecule for r in records], ["water", "co2"])


class TestSelection(unittest.TestCase):
    def setUp(self):
        self.result = parse_molpro_xml(os.path.join(data_dir, "multi_step.xml"))
        name_molecules(self.result)

    def select(self, **kwargs):
        return select_records(self.result, ExtractionSpec(**kwargs))

    def test_default_selection(self):
        selected = self.select()
        # RHF energy, CMRCC total and excitation energy, and only the last RKS energy of the geometry optimization
        self.assertEqual([r.value for r in selected], [-38.0, -38.5, 0.25, -1.5])

    def test_nested_steps(self):
        self.assertEqual([r.value for r in self.select(nested_steps="none")], [-38.0, -38.5, 0.25])
        self.assertEqual([r.value for r in self.select(nested_steps="all")], [-38.0, -38.5, 0.25, -1.0, -1.5])
        with self.assertRaises(ValueError):
            ExtractionSpec(nested_steps="first")

    def test_property_selection(self):
        self.assertEqual([r.property for r in self.select(properties=("exc. energy",))], ["Exc. energy"])
        # None selects everything that is principal
        self.assertEqual(len(self.select(properties=None)), 5)
        self.assertEqual(len(self.select(properties=None, principal_only=False, nested_steps="all")), 7)

    def test_scf_suffix_is_only_stripped_if_not_in_input(self):
        self.assertEqual(self.select()[0].command, "DF-RHF")
        self.result.inputs[0] += "\n{df-rhf-scf}"
        self.assertEqual(self.select()[0].command, "DF-RHF-SCF")
        self.assertEqual(self.select(strip_scf_suffix=False)[0].command, "DF-RHF-SCF")

    def test_dispersion_correction(self):
        result = parse_molpro_xml(os.path.join(data_dir, "dft_dispersion.xml"))
        selected = select_records(result)
        self.assertEqual([(r.property, r.method) for r in selected], [("Energy", "RKS+D4")])
        selected = select_records(result, ExtractionSpec(merge_dispersion=False))
        self.assertEqual(selected[0].method, "RKS")
        # The correction is available as a property of its own
        selected = select_records(result, ExtractionSpec(properties=("dispersion correction",), principal_only=False))
        self.assertEqual(selected[0].method, "PBE D4")

    def test_errored_steps_are_skipped(self):
        for r in self.result.records:
            if r.command == "DF-CMRCC":
                r.errored = True
        self.assertEqual([r.value for r in self.select()], [-38.0, -1.5])
        self.assertEqual(len(self.select(skip_errored_steps=False)), 4)

    def test_no_label_by_default(self):
        self.assertEqual({r.label for r in self.select()}, {None})

    def test_fixed_label(self):
        self.assertEqual({r.label for r in self.select(label="CAS(3,3)")}, {"CAS(3,3)"})
        self.assertEqual({r.label for r in self.select(label="")}, {None})

    def test_label_from_path(self):
        self.result.path = "/data/active_space/run.1-2.xml"
        self.assertEqual({r.label for r in self.select(label_from_path=r"/([^/]+)/[^/]+$")}, {"active_space"})
        # Without capture group the whole match is used
        self.assertEqual({r.label for r in self.select(label_from_path="old_data|active_space")}, {"active_space"})
        self.assertEqual({r.label for r in self.select(label_from_path="old_data")}, {None})

    def test_label_from_input(self):
        self.assertEqual({r.label for r in self.select(label_from_input=r"^basis=(\S+)")}, {"cc-pVDZ"})
        self.assertEqual({r.label for r in self.select(label_from_input=r"nothing")}, {None})
        # A capture group that does not take part in the match gives no label
        self.assertEqual({r.label for r in self.select(label_from_input=r"(nothing)?basis")}, {None})

    def test_label_parts_are_combined(self):
        self.result.path = "/data/ref/run.1-2.xml"
        selected = self.select(label="x", label_from_path="ref", label_from_input=r"basis=(\S+)")
        self.assertEqual({r.label for r in selected}, {"x | ref | cc-pVDZ"})

    def test_label_is_per_job(self):
        self.result.records[0].job_index = 1
        self.result.inputs[1] = "basis=vtz"
        labels = {(r.job_index, r.label) for r in self.select(label_from_input=r"basis=(\S+)")}
        self.assertEqual(labels, {(0, "cc-pVDZ"), (1, "vtz")})

    def test_invalid_regular_expression(self):
        with self.assertRaises(ValueError):
            ExtractionSpec(label_from_path="(")

    def test_records_are_not_modified(self):
        self.select()
        self.assertEqual(self.result.records[0].command, "DF-RHF-SCF")


NO_INPUT_XML = """<?xml version="1.0"?>
<molpro xmlns="http://www.molpro.net/schema/molpro-output"><job>
 <jobstep command="DF-RHF-SCF"><property name="Energy" method="RHF" principal="true" value="-1.0"/></jobstep>
</job></molpro>
"""


class TestInputFileFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def touch(self, name, content=""):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_prefix_match(self):
        xml = self.touch("calc.20261119-130642-ab12.xml", NO_INPUT_XML)
        inp = self.touch("calc.inp", "basis=vdz\n  geometry=water.xyz\n")
        self.assertEqual(find_input_file(xml), inp)
        result = parse_molpro_xml(xml)
        self.assertEqual(result.inputs[0], "basis=vdz\ngeometry=water.xyz")
        self.assertEqual(result.problems, [])
        name_molecules(result)
        self.assertEqual(result.records[0].molecule, "water")

    def test_exact_match_and_longest_prefix(self):
        xml = self.touch("calc_dz.1-2.xml")
        self.touch("calc.inp")
        self.touch("calc_dz.inp")
        exact = self.touch("calc_dz.1-2.inp")
        self.assertEqual(find_input_file(xml), exact)
        os.remove(exact)
        self.assertEqual(find_input_file(xml), os.path.join(self.tmp.name, "calc_dz.inp"))

    def test_no_false_matches(self):
        xml = self.touch("calc_dz_gecco.1-2.xml")
        # Only '.' separates the base name from the scheduler ID; similar but different names must not match
        self.touch("calc.inp")
        self.touch("calc_dz_gecco_other.inp")
        self.assertIsNone(find_input_file(xml))

    def test_missing_input_is_reported(self):
        xml = self.touch("calc.xml", NO_INPUT_XML)
        result = parse_molpro_xml(xml)
        self.assertEqual(result.inputs, {})
        self.assertEqual(len(result.problems), 1)

    def test_input_in_xml_takes_precedence(self):
        xml = os.path.join(data_dir, "multi_step.xml")
        self.assertIn("geometry=/some/path/ch.xyz", parse_molpro_xml(xml).inputs[0])


if __name__ == "__main__":
    unittest.main()
