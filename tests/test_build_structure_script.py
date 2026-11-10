#!/usr/bin/env python3

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest

from koehnlab.molecular import readLabeledMolecule

import numpy as np
from numpy.testing import assert_almost_equal

script_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(script_dir, "molecules")

# Load the script as module (the scripts directory is not a package)
_spec = importlib.util.spec_from_file_location(
    "build_structure", os.path.join(script_dir, "..", "scripts", "build_structure.py")
)
assert _spec is not None and _spec.loader is not None
build_structure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_structure)


class TestBuildStructureScript(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def write(self, name: str, content: str) -> str:
        with open(self.path(name), "w") as f:
            f.write(content)
        return self.path(name)

    def run_script(self, *args: str):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = build_structure.main(list(args))
            except SystemExit as exit:  # argparse errors
                code = exit.code
        return code, err.getvalue()

    def test_water(self):
        spec = self.write("water.txt", "O1 - H1 0.96\nO1 - H2 0.96\nH1 - O1 - H2 104.5\n")
        output = self.path("water.xyz")

        code, err = self.run_script("-i", spec, "-o", output, "--labels", "-v")

        self.assertEqual(code, 0)
        self.assertIn("H1 - O1 - H2", err)
        molecule, labels = readLabeledMolecule(output)
        self.assertEqual(labels, ["O1", "H1", "H2"])
        assert_almost_equal(np.linalg.norm(molecule.coordinates()[1] - molecule.coordinates()[0]), 0.96, decimal=6)

        # Without --labels, plain element symbols are written
        plain = self.path("plain.xyz")
        self.assertEqual(self.run_script("-i", spec, "-o", plain)[0], 0)
        with open(plain) as f:
            self.assertEqual(f.read().split("\n")[2].split()[0], "O")

    def test_guess_and_planar(self):
        spec = self.write("spec.txt", "C1 - C2 1.40\n")
        output = self.path("out.xyz")

        code, _ = self.run_script(
            "-i", spec, "-o", output, "--guess", os.path.join(data_dir, "biphenyl_aligned.xyz"), "--planar"
        )

        self.assertEqual(code, 0)
        molecule, _ = readLabeledMolecule(output)
        self.assertEqual(molecule.nAtoms(), 22)
        assert_almost_equal(molecule.coordinates()[:, 2], np.zeros(22), decimal=8)

    def test_directive_in_file_and_linear_flag(self):
        spec = self.write("co2.txt", "O1 - C1 1.16\nC1 - O2 1.16\n")
        output = self.path("co2.xyz")

        self.assertEqual(self.run_script("-i", spec, "-o", output, "--linear")[0], 0)
        molecule, _ = readLabeledMolecule(output)
        assert_almost_equal(molecule.coordinates()[:, :2], np.zeros((3, 2)), decimal=10)

    def test_symmetry(self):
        spec = self.write(
            "nh3.txt",
            "N1 - H1 1.01\nN1 - H2\nN1 - H3\nH1 - N1 - H2 106.7\n"
            "@symmetry C3(z): H1->H2->H3\n@symmetry sigma(y): H2<->H3\n",
        )
        output = self.path("nh3.xyz")

        self.assertEqual(self.run_script("-i", spec, "-o", output, "--labels")[0], 0)
        molecule, labels = readLabeledMolecule(output)
        distances = [
            np.linalg.norm(molecule.coordinates()[labels.index(a)] - molecule.coordinates()[labels.index("N1")])
            for a in ("H1", "H2", "H3")
        ]
        assert_almost_equal(distances, np.full(3, 1.01), decimal=6)

        # Symmetric structures are always oriented
        code, err = self.run_script("-i", spec, "-o", output, "--no-orient")
        self.assertEqual(code, 2)
        self.assertIn("orient", err)

    def test_failure_exit_codes(self):
        # inconsistent parameters
        spec = self.write("bad.txt", "C1 - C2 1.0\nC2 - C3 1.0\nC1 - C3 5.0\n")
        code, err = self.run_script("-i", spec, "-o", self.path("o.xyz"), "--max-attempts", "1")
        self.assertEqual(code, 1)
        self.assertIn("not satisfied", err)

        # invalid input
        spec = self.write("invalid.txt", "C1 - C2 abc\n")
        code, err = self.run_script("-i", spec, "-o", self.path("o.xyz"))
        self.assertEqual(code, 2)
        self.assertIn("invalid.txt:1", err)

        # missing file
        code, _ = self.run_script("-i", self.path("missing.txt"), "-o", self.path("o.xyz"))
        self.assertEqual(code, 2)

        # contradicting options
        self.assertNotEqual(
            self.run_script("-i", spec, "-o", self.path("o.xyz"), "--planar", "--linear")[0], 0
        )


if __name__ == "__main__":
    unittest.main()
