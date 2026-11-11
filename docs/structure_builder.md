# Structure builder

The structure builder turns a set of **structure parameters** (bond lengths, bond angles and dihedral angles) into an XYZ
structure. The parameters can be whatever is convenient: they do **not** have to form a Z-matrix, they may be redundant, and they
may be incomplete.

- [Quick start](#quick-start)
- [How it works](#how-it-works)
- [The parameter file](#the-parameter-file)
- [Starting geometry](#starting-geometry)
- [Planar and linear molecules](#planar-and-linear-molecules)
- [Symmetry](#symmetry)
- [Output and orientation](#output-and-orientation)
- [Command line reference](#command-line-reference)
- [Python API](#python-api)
- [Troubleshooting](#troubleshooting)
- [Limitations](#limitations)

All example files used in this document are in [`examples/`](examples) and are built by the test suite.


## Quick start

Write the parameters into a text file, one per line:

```
# water.txt
O1 - H1 0.96
O1 - H2 0.96
H1 - O1 - H2 104.5
```

and run

```bash
build_structure.py --input water.txt --output water.xyz
```

(the script is in `scripts/`; the `koehnlab` package has to be on your `PYTHONPATH`, see the main [README](../README.md)).
Lengths are in Angstrom and angles in degrees. Use `--verbose` to see how well every parameter is fulfilled:

```
parameter                          target     achieved    deviation
O1 - H1                          0.960000     0.960000    -1.11e-16
O1 - H2                          0.960000     0.960000    -1.11e-16
H1 - O1 - H2                   104.500000   104.500000     0.00e+00
closest atoms: O1 - H1             > 0.60     0.960000
```


## How it works

Every parameter is an equation for the Cartesian coordinates of the atoms involved, for example `|r_O - r_H1| = 0.96`. A set of
parameters is therefore a (non-linear) system of equations, and any solution is a structure that has these parameters. The builder
solves the system with a least-squares optimization (trust-region algorithm of `scipy.optimize.least_squares`). Each residual only depends on a few
atoms, so the derivatives are computed blockwise and vectorized (by central differences), which keeps larger molecules fast.

This has a few consequences:

- **Any set of parameters works**, as long as it is consistent. Parameters may be redundant (for example all angles of a ring),
  but if they contradict each other, the build does not converge and the report shows which parameters are not fulfilled.
- **Translation and rotation are arbitrary** (they do not change any parameter), which is why the result is always brought
  into a [standard orientation](#output-and-orientation).
- **Missing parameters leave freedom.** What the parameters do not determine stays close to the *starting geometry*: all bonds
  and angles that are not specified, and the torsions about all bonds without a specified dihedral angle, are weakly restrained to
  their values in the starting geometry. The result is thus the solution closest to the start. Without a starting geometry, a rough
  one is [generated from the connectivity](#the-default-starting-geometry).
- **There can be several solutions** (mirror images, cis/trans, folded rings, ...). The one closest to the starting geometry is
  found, so a good starting geometry matters for flexible or ring-containing molecules where the parameters are not sufficient.

The following requirements are always enforced:

- **No two atoms come closer than 0.6 Angstrom.** This never matters for sensible structures but prevents atoms from lying on
  top of each other, which can satisfy parameters that do not determine the structure. It is weighted much more strongly than the
  parameters: if the parameters cannot be fulfilled together with it, the parameters give way and are reported as not satisfied.
- **No two atoms come closer than 0.75 times the sum of their covalent radii** (e.g. 0.80 Angstrom for C-H). This does not apply
  to a pair whose distance is specified explicitly, nor to pairs that are equivalent to it by the symmetry operations (only the
  0.6 Angstrom limit applies there). The shortest ordinary bonds (triple bonds such as N-N, ratio about 0.78) stay above the
  threshold. Bonds that are shorter, e.g. multiple bonds between metals such as Cr-Cr, have to be specified explicitly.
- A weak repulsion keeps non-bonded atoms (and atoms bonded to a common neighbor) from coming too close to each other during the
  optimization.

A structure counts as **converged** if every bond length is within 1e-4 Angstrom of its target, every angle and dihedral angle within
0.01 degrees, and the minimum distance is respected. If an attempt does not converge, the starting coordinates are randomly
displaced and the optimization is repeated (up to `--max-attempts` times, default 5, reproducible via `--seed`). The best result is
returned either way.


## The parameter file

The file contains one record per line. Everything after `#` is a comment, empty lines are ignored.

```
C1 - C2 1.34                # bond length (Angstrom); also states that C1 and C2 are bonded
C1 - H1                     # no value: only states that the atoms are bonded
C1 - C2 - C3 123.5          # angle (degrees), the atom in the middle is the apex
C1 - C2 - C3 - C4 180       # dihedral angle (degrees)
C1 - C2 1.34 w=2            # optional weight of a parameter relative to the others (default 1)
```

**Atoms** are identified by labels consisting of an element symbol and a number: `C1`, `H12`, `Cl2`. The element is taken from the
label, so there is no separate list of atoms. The atoms are numbered in order of their first appearance in the file (this is the order
in the output).

**The number of labels** determines the type of a record: two labels are a bond, three an angle and four a dihedral angle.

- Two-label records always declare a bond. The bonds define the molecule's *connectivity*, which is needed for the starting
  geometry, the restraints and the [symmetry inference](#inferring-missing-atoms). Every atom must be connected to the others through
  bonds, so hydrogen atoms need bond records (`C1 - H1`) when no [starting geometry](#starting-geometry) is given.
- Angles and dihedral angles need not be along bonds. They must have a value.
- **Dihedral angles** follow the IUPAC convention: looking along the bond from the second to the third atom, the angle is positive
  if the bond to the first atom has to be rotated clockwise to eclipse the bond to the fourth atom. Values are wrapped into
  (-180, 180]. Mirror images have opposite signs, so a set of dihedral angles fixes the handedness.
- Each parameter can only be given once. The reversed form (`C2 - C1`, `C3 - C2 - C1`, `C4 - C3 - C2 - C1`) is the same parameter;
  repeating it with the same value is harmless, with a different value it is an error.

**Directives** start with `@` and apply to the whole molecule: `@planar`, `@linear` (see
[Planar and linear molecules](#planar-and-linear-molecules)), `@symmetry` and `@group` (see [Symmetry](#symmetry)).

Errors in the file are reported with the line number, for example `spec.txt:3: Invalid number 'abc'`.

### Example: butadiene

[`examples/butadiene.txt`](examples/butadiene.txt) specifies half of the structure and leaves the rest (the hydrogen positions)
to the defaults:

```
@planar

C1 - C2 1.34
C3 - C4 1.34
C2 - C3 1.46
C1 - C2 - C3 123.5
C2 - C3 - C4 123.5
C1 - C2 - C3 - C4 180

C1 - H1
C1 - H2
C2 - H3
C3 - H4
C4 - H5
C4 - H6
```

### The default starting geometry

Without a starting geometry, the builder places the atoms one after another along a spanning tree of the bonds. Values from the
parameter file are used wherever they belong to the placement of an atom; everything else is taken from simple defaults:

| Quantity | Default |
| --- | --- |
| bond length | sum of the covalent radii |
| angle at an atom with 2 or 3 bonds | 120 degrees |
| angle at an atom with 4 bonds | 109.47 degrees |
| angle at an atom with more bonds | 90 degrees |
| torsions | substituents distributed evenly around the bond (staggered / planar) |
| torsions along a ring | cis, so that the ring continues in the same direction and closes; substituents of ring atoms point away from the ring |

The defaults are deliberately simple. Linear arrangements are never assumed (the torsion about a linear fragment is undefined), so
specify angles of 180 degrees explicitly. For anything chemically more involved (lone pairs, conjugation, metal complexes, ...)
provide a starting geometry.


## Starting geometry

A starting geometry is passed with `--guess file.xyz`. Everything the parameters do not specify stays close to it, and it
replaces the default geometry (its quality matters most for rings and for molecules with many torsions).

The XYZ file can use plain element symbols, in which case the atoms are labeled per element in order (`C1, C2, ..., H1, H2, ...`), or
labels directly in the element column:

```
8
comment line
C1  0.000000  0.000000  0.765000
C2  0.000000  0.000000 -0.765000
H1  0.883346  0.510000  1.160000
...
```

(mixing both is an error). The labels are what the parameter file refers to. The atoms of the output are in the order of the starting
geometry, and the parameter file may mention only some of them.

- The **connectivity** is the union of the bonds in the parameter file and the bonds detected in the starting geometry (two atoms
  are bonded if their distance is less than 1.25 times the sum of their covalent radii). Hydrogens therefore do not need bond
  records.
- Elements in the parameter file must match the geometry, and every label of the parameter file must exist in it.
- Symmetry operations and planarity refer to the coordinate system of the starting geometry, see
  [Symmetry](#symmetry) and [Planar and linear molecules](#planar-and-linear-molecules).


## Planar and linear molecules

`@planar` (or `--planar`) restricts all atoms to one plane, `@linear` (or `--linear`) to one line. This is exact: the optimization only
varies the allowed coordinates, so planarity is not a penalty. The plane is the xy plane and the line the z axis; the standard
orientation of the result keeps this (planar molecules end up in the xy plane, linear molecules along z).

**Planar molecules**

- Dihedral angles can only be 0 or 180 degrees (cis or trans); other values are rejected.
- Cis/trans is a discrete choice, which the optimization cannot change continuously. The choice is made when the starting
  geometry is generated or given, and a dihedral angle that is not realized is reported as not satisfied. For rings, the builder
  first fulfills the parameters without the restriction to a plane, which lets the ring close, and then projects onto the plane.
- A starting geometry is projected onto its best-fit plane, with a warning if it is far (more than 0.05 Angstrom rms) from planar.

**Linear molecules**

- The molecule must be a chain (every atom has at most two bonds, no rings), all specified angles must be 180 degrees, and
  dihedral angles cannot be specified.
- Bond lengths determine the structure. [`examples/co2.txt`](examples/co2.txt) is just
  `@linear` plus the two bond lengths.


## Symmetry

If the molecule has symmetry, it can be imposed exactly. Parameters that are equivalent by symmetry are then tied together, so it is
enough to specify one of them, and the result has exactly the requested symmetry. Conflicting values for equivalent parameters are
rejected (`The parameters ... are equivalent by symmetry but have different values`).

A symmetry operation is described by

1. the operation itself (a matrix), and
2. which atoms it exchanges.

### Operations

```
@symmetry <operation>: <atoms>
```

| Operation | Meaning |
| --- | --- |
| `i` | inversion through the origin |
| `sigma(axis)` | mirror plane through the origin, perpendicular to the axis |
| `Cn(axis)` | rotation by 360/n degrees about the axis (right-handed: counterclockwise seen from the positive end of the axis) |
| `Sn(axis)` | rotation by 360/n degrees about the axis, followed by a reflection at the plane perpendicular to it |

The axis is `x`, `y`, `z` or a vector `a,b,c` (for example `C2(1,1,0)`). All operations of a molecule refer to **the same coordinate
system, and its origin has to lie on every symmetry element**. For a starting geometry, this is the coordinate system of the file.

The atoms are listed as comma-separated entries:

| Entry | Meaning |
| --- | --- |
| `A<->B` | the operation exchanges A and B |
| `A->B->C` | the operation moves A to the position of B, B to the position of C and C to the position of A |
| `A` | A stays where it is (it lies on the symmetry element) |
| `...` | the list is incomplete, the remaining atoms are [inferred](#inferring-missing-atoms) |

Atoms that are not listed stay where they are, *unless* the list contains `...`. For a complete list, make sure every atom that is
moved by the operation is listed; an atom that is forgotten is assumed to lie on the symmetry element, which usually violates
the bonds and is reported (`... moves the bond A - B to ..., which is not a bond (are atoms missing in the operation?)`).

Example, [`examples/ammonia.txt`](examples/ammonia.txt):

```
N1 - H1 1.01
N1 - H2
N1 - H3
H1 - N1 - H2 106.7

@symmetry C3(z): H1->H2->H3
@symmetry sigma(y): H2<->H3
```

Only one N-H bond and one H-N-H angle are given; the others follow from the C3v symmetry. The operations are the generators of the
symmetry group; the builder creates the whole group from them (at most 192 operations; a group that is not finite is rejected).

The builder checks that the operations are consistent: they have to preserve elements and bonds, no operation may exchange atoms
without moving them, and the values of equivalent parameters must agree. Under operations that include a reflection or inversion,
dihedral angles change sign (a dihedral angle is then only consistent if it is 0 or 180 degrees, or if the equivalent dihedral
has the opposite value).

### Point groups

`@group <name>` adds the generators of a point group in a canonical frame, with all atoms to be inferred:

| Group | Generators |
| --- | --- |
| `Cs` | `sigma(z)` |
| `Ci` | `i` |
| `Cn` | `Cn(z)(n is a number.) The table lists the canonical frame: the principal axis along z, the perpendicular C2 axes along x, the vertical
mirror plane `sigma(y)` in the xz plane and the horizontal one in the xy plane. The frame adapts to the `@symmetry` lines you give
for the same molecule: it is rotated (by permuting the axes) such that all of your operations are elements of the group. For
example, with `@group C2v` and `@symmetry C2(x): H1<->H2, ...`, the principal axis becomes x. If several orientations fit (here, which
of the two mirror planes is `sigma(y)`), the one that moves the fewest axes is used; give one more operation to choose. Without
any `@symmetry` line, or if no axis permutation fits, the canonical frame is used, and the molecule or starting geometry has to be in it.
A planar molecule (`@planar` or `--planar`) lies in the xy plane, so the frame is also rotated such that the xy plane is a mirror plane of the group:
with `@group C2v` and `@planar`, the C2 axis lies in the plane (x or y).

 Oh |
| `I`, `Ih` | `C2(z)`, `C3(1,1,1)`, `C5(0,1,phi)` (phi: golden ratio), plus `i` for Ih |

(n is a number.) The canonical frame has the principal axis along z, the perpendicular C2 axes along x, the vertical mirror plane
`sigma(y)` in the xz plane and the horizontal one in the xy plane. A molecule or starting geometry that is described with `@group` has
to be in this frame. If your molecule has a different orientation (like the benzyne example below), use explicit `@symmetry` lines
instead.

### Inferring missing atoms

Operations whose atoms are incomplete (`...`, or generated by `@group`) are completed automatically. The builder looks for
all atom permutations that

- map the bonds onto bonds and atoms onto atoms of the same element,
- contain the atoms you gave, and
- have cycle lengths that are compatible with the order of the operation (a rotation by 120 degrees can only move atoms in
  cycles of length 1 or 3).

If exactly one permutation remains, it is used. If there are several, a starting geometry resolves the ambiguity: the permutation that
the operation maps best onto the geometry is chosen (if that clearly stands out). Otherwise you get an error that shows two of the
candidates and asks for more atoms or a starting geometry:

```
The symmetry operation sigma(y) is ambiguous: 2 atom mappings are compatible with the bonds and the given atoms (e.g. 'all atoms fixed' or
'H1->H2, H2->H1'). Specify more atoms of the operation.
```

If a molecule has no valid mapping at all, the error names the atom where the search breaks down, which usually points to a wrong bond
in the file:

```
No atom mapping exists for the symmetry operation C2(y) that is compatible with the bonds and the given atoms. The mapping breaks down at H5
(H, 1 bond: C1): C1 is moved to C2, so H5 would have to be moved to an atom of the same element bonded to C2, but H2 has different bonds (H2: 2 bonds:
C2, C4). Check for missing or extra bonds around these atoms
```

You can mix `@group` with explicit lines: a `@symmetry` line for an operation of the group (compared by its matrix, so
`sigma(0,1,0)` is the same as `sigma(y)`) provides the atoms you know, and the rest is inferred. Example,
[`examples/ethane_d3d.txt`](examples/ethane_d3d.txt) together with [`examples/ethane_guess.xyz`](examples/ethane_guess.xyz):

```
C1 - C2 1.53
C1 - H1 1.09
H1 - C1 - C2 111

@group D3d
```

```bash
build_structure.py -i ethane_d3d.txt -g ethane_guess.xyz -o ethane.xyz
```

Three parameters determine the whole molecule. The (rough, but symmetric and correctly oriented) starting geometry selects which of the
automorphisms of the bond graph are the symmetry operations.

### Symmetry and planarity

`@planar` is a mirror plane (the xy plane) that leaves all atoms in place, so it can be combined with symmetry operations.
For a planar molecule the plane is the xy plane of the *symmetry frame*, so axes in the molecular plane are x or y (or a vector in the
xy plane) and axes perpendicular to it are z. [`examples/benzyne.txt`](examples/benzyne.txt):

```
@planar
...
@symmetry C2(y): C1<->C2, C3<->C6, ...
@symmetry sigma(x): C1<->C2, C3<->C6, ...
```

The C2 axis runs through the middle of the C1-C2 and C4-C5 bonds (the y axis), and the second mirror plane is perpendicular to the
molecule (normal along x). A common mistake is to use `C2(z)` for a planar molecule: the z axis is perpendicular to the molecular plane
there, and `sigma(z)` is the plane of the molecule itself, which would put exchanged atoms on top of each other.

### Without a starting geometry

The default geometry is not symmetric. The builder therefore first relaxes it with a penalty for violating the symmetry and then
restricts the structure exactly to the symmetric ones. This works well when the symmetric structure is close to what the
defaults generate; for unusual structures provide a starting geometry in the symmetry frame (a warning is issued if it deviates from the
symmetry by more than 0.05 Angstrom rms; it is projected onto the symmetric structures).


## Output and orientation

The result is written as XYZ file. With `--labels`, the element column contains the labels (`C1`, `H3`, ...) instead of the plain
element symbols, which can be read back with `readLabeledMolecule` and used as a starting geometry.

Because translations and rotations are arbitrary, the result is brought into a **standard orientation** (this is always done for
structures with symmetry; `--no-orient` disables it for all others):

- The geometric center (the average of all atomic positions) is the origin. All atoms are weighted equally, so isotopes do not
  matter.
- The principal axes of the (unweighted) inertia tensor are the coordinate axes, ordered by increasing moment of inertia.
- The signs and the orientation of the axes are chosen deterministically, using the order of the atoms: for an asymmetric top, the
  first atom that is not in the nodal plane of an axis has a positive coordinate along it; a right-handed frame is always used, so the
  molecule is **never mirrored**. For a symmetric top the unique axis is z and the first atom off this axis lies in the xz plane (with
  positive x); for a spherical top, z points to the first atom off the center and the first atom off that axis lies in the xz plane; for
  a linear molecule, the molecule is along z.
- The same function is available as `Molecule.bringToStandardOrientation()`, also for the command line script
  `molecule_processor.py --action orient` (with `--center mass` for the center of mass and mass-weighted inertia tensor).

The rigid rotation to the standard orientation preserves all parameters and the symmetry of the structure (only the frame
changes).


## Command line reference

```
build_structure.py -i PARAMETERS -o OUTPUT [options]
```

| Option | Description |
| --- | --- |
| `-i`, `--input PATH` | parameter file |
| `-o`, `--output PATH` | XYZ file to write |
| `-g`, `--guess PATH` | XYZ file with a [starting geometry](#starting-geometry) |
| `--planar` / `--linear` | same as the directives `@planar` / `@linear` |
| `--no-orient` | don't bring the result into the standard orientation (not possible with symmetry) |
| `--labels` | write atom labels instead of element symbols |
| `--max-attempts N` | number of attempts with randomly displaced starting coordinates (default 5) |
| `--seed N` | seed of the random displacements (default 0) |
| `-v`, `--verbose` | print how well every parameter is fulfilled |

The exit code is `0` if all parameters are fulfilled, `1` if the optimization did not fulfill them (the table of parameters is printed
in this case, and the structure is still written), and `2` for invalid input (file errors, symmetry that does not fit the molecule, ...).

`build_structure.py --help` prints a summary of the file format.


## Python API

Everything is available from `koehnlab.molecular`:

```python
from koehnlab.molecular import (
    parseStructureSpec,       # read a parameter file (parseStructureSpecString for a string)
    buildStructure,
    readLabeledMolecule,      # XYZ file with labels -> (Molecule, labels)
    writeMolecule,
)

spec = parseStructureSpec("water.txt")
result = buildStructure(spec)

print(result.converged)
print(result.format())                 # table of the parameters, as printed with --verbose
writeMolecule(result.molecule, "water.xyz", labels=result.labels)
```

`buildStructure(spec, guess=None, guessLabels=None, orient=True, maxAttempts=5, seed=0, geometry=None)`

- `guess`: a `Molecule` as starting geometry; `guessLabels` its labels (default: per-element counters).
- `geometry`: `"planar"` or `"linear"`, overriding the directives of the specification.
- `orient`: must stay `True` for structures with symmetry (a `ValueError` is raised otherwise).
- Returns a `BuildResult`:
  - `molecule`: the structure (a `Molecule`),
  - `labels`: label of every atom,
  - `converged`: all parameters fulfilled and the minimum distance respected,
  - `attempts`: number of attempts that were needed,
  - `parameters`: list of `ParameterReport` (labels, target, achieved, deviation) for every specified parameter,
  - `closestAtoms`, `minimumDistance`: the closest pair of atoms.

Invalid specifications raise `StructureSpecError` (a subclass of `ValueError`); the parser messages contain file name and line number.

Further building blocks, which can be used on their own:

| Function | Purpose |
| --- | --- |
| `defaultGuess(spec, geometry=None)` | the rough geometry generated from the connectivity |
| `koehnlab.molecular.InternalCoordinates` | bond lengths, angles and dihedral angles of coordinates (`bondLengths`, `bondAngles`, `dihedralAngles`) |
| `pointGroupGenerators(name)` | generators (name and matrix) of a point group in the canonical frame |
| `adaptedPointGroupGenerators(name, matrices)` | the same, in the frame (axis permutation) in which the given operations belong to the group |
| `Molecule.bringToStandardOrientation(center="geometric")` | the orientation described [above](#output-and-orientation) |


## Troubleshooting

| Message / symptom | Cause |
| --- | --- |
| `... Invalid number ...`, `Expected atom labels ...` | syntax error in the parameter file, the message contains the line |
| `The atoms are not all connected via bonds` | an atom (typically a hydrogen) is missing from the bond records; add `C1 - H1`-style records or provide a starting geometry |
| `No atom mapping exists for the symmetry operation ...` | the operation does not fit the bonds. The message names the atom where the mapping breaks down: look for missing or extra bonds around it, or an operation that does not match the molecule's orientation |
| `The symmetry operation ... is ambiguous` | give more atoms of the operation (`H1` alone means "stays", `A<->B` an exchange) or a starting geometry |
| `... moves the bond A - B to ..., which is not a bond` | an operation with a complete list leaves out atoms (unlisted atoms stay where they are); add them or `...` |
| `... are equivalent by symmetry but have different values` | the same parameter was given with different values for symmetry-equivalent atoms |
| `Dihedral angle ... is not possible in a planar molecule` | only 0 and 180 degrees are possible with `@planar` |
| exit code 1, `<-- not satisfied` in the table | the parameters are inconsistent (or the build ended in a bad local solution). Check the listed parameters, try another `--seed` or more `--max-attempts`, or provide a starting geometry |
| `<-- too close` | the parameters would put two atoms closer than 0.6 Angstrom |
| warning `The starting geometry is far from planar` / `deviates from the symmetry` | the starting geometry is projected onto the plane or the symmetric structures; check that it is given in the right frame (symmetry elements through the origin) |

If a structure converges but looks wrong, the parameters probably do not determine it: other solutions (folded rings, flipped
substituents) satisfy them as well. Add the missing parameters or provide a starting geometry.

For small molecules, setting `OMP_NUM_THREADS=1` can make the build several times faster (the linear algebra of the solver is
faster single-threaded for small matrices).


## Limitations

- Hydrogen atoms (and all other atoms) must be connected by bond records unless a starting geometry is given.
- Only bonds, angles and dihedral angles can be specified. Distances between non-bonded atoms are not supported.
- The default starting geometry only knows the number of bonds of an atom, not hybridization or lone pairs.
- In planar molecules, cis/trans is a discrete property that the optimization cannot change; unrealized dihedral angles are reported.
- Default bond lengths and the detection of bonds in a starting geometry use the covalent radii of ASE, which are meaningful up to curium (Z = 96).
- Symmetry operations are given by the user (the `@group` frame is canonical unless your `@symmetry` lines imply another one); the symmetry of a molecule is not detected automatically.
