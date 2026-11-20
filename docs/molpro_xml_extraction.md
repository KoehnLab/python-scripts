# Extracting results from Molpro XML files

`scripts/extract_data.py` collects the results of Molpro calculations from their XML output files (`*.xml`) in one table. It works for any
number of files, molecules, basis sets, methods and states.

## Quick start

```bash
# Search directories recursively (or pass XML files directly) and print a table
extract_data.py path/to/calculations

# Write a file; the format follows the suffix (.csv, .tsv, .json, .jsonl, .md, .parquet, .xlsx)
extract_data.py path/to/calculations -o results.csv

# One column per basis set, e.g. to compare excitation energies
extract_data.py path/to/calculations -p "Exc. energy" --pivot basis_set
```

Example output:
```
molecule basis_set  command     property   method symmetry state value unit
      ch   cc-pVDZ   DF-RHF       Energy      RHF        1     1 -38.0  E_h
      ch   cc-pVDZ DF-CMRCC Total energy    MRCC2        1     1 -38.5  E_h
      ch   cc-pVDZ DF-CMRCC  Exc. energy LR-MRCC2        2     1  0.25  E_h
   water DEF2-TZVP   DF-RKS       Energy   RKS+D4        1     1 -76.4  E_h
```

Warnings and errors that Molpro wrote to the XML files, as well as files that could not be read, are listed on stderr (silence with `-q`).

## The table

The result is a *tidy* table with one row per extracted value. All aspects of a value are columns:

| Column | Meaning |
|---|---|
| `molecule` | Name of the XYZ file given via `geometry=<name>.xyz` in the input. Otherwise the Hill formula of the geometry (e.g. `C3H5`), otherwise the name of the XML file without the run suffix |
| `formula` | Hill formula, if the file contains the geometry (only reported by steps that write it, e.g. SCF) |
| `basis_set`, `functional` | From the `<summary>` of the job step; steps without a summary inherit the value of earlier steps |
| `command` | Molpro command of the job step. A trailing `-SCF` is dropped (`DF-RHF-SCF` -> `DF-RHF`) unless the input uses the long name |
| `property`, `method` | Name and method of the property, e.g. `Exc. energy` / `LR-MRCC2`. For energies of calculations with a dispersion correction the method gets a suffix, e.g. `RKS+D4` |
| `multiplicity`, `symmetry`, `state` | State labels as reported by Molpro |
| `value`, `unit` | Value (empty for non-scalar properties such as dipole moments) and its unit (atomic units) |
| `principal`, `messages` | Whether Molpro flags the property as principal; warnings/errors of the job step |
| `source_path`, `job_index`, `step_index` | Where the value came from |

If an XML file does not contain the input, the input file next to it is used instead: `<name>.inp` (or `.com`) in the same directory, where `<name>` is the base name of the XML file or a prefix of it ending before a `.`, so `calc.inp` belongs to `calc.20261119-130642-ab12.xml`. The longest matching name wins.

Several molecules or basis sets in a single calculation are supported: the context is tracked per job step, and several `geometry=` entries in
the input are assigned to the geometries in the order they occur.

## Choosing what to extract

| Option | Effect |
|---|---|
| `-p NAME ...` / `--all-properties` | Properties to extract (default: `Energy`, `Total energy`, `Exc. energy`) |
| `--include-non-principal` | Also take properties that are not flagged as principal |
| `--nested-steps none\|last\|all` | Steps inside other steps, e.g. the single points of a geometry optimization (default: the last one) |
| `--keep-errored` | Do not skip job steps that reported an error |
| `--columns ...` / `--all-columns` | Columns to output |
| `--pivot COLUMN` | Spread the values of a column over several columns (the first value is kept if results are not unique) |

## Using it from Python

The script is a thin wrapper around the `koehnlab.io` modules. These need `pandas` and `lxml`, and can be used directly (e.g. in a notebook):

```python
from koehnlab.io.molpro_table import extract_table
from koehnlab.io.molpro_extract import ExtractionSpec

extraction = extract_table(["path/to/calculations"], ExtractionSpec(properties=("Exc. energy",)))
table = extraction.table              # pandas.DataFrame
extraction.problems                   # [(path, description), ...]

table.pivot_table(index=["molecule", "state"], columns="basis_set", values="value", observed=True)
```

| Module | Purpose |
|---|---|
| `molpro_xml` | Parser. Returns plain `Record` objects, no pandas needed. Also names molecules |
| `molpro_extract` | `ExtractionSpec` and `select_records`: which results to keep and how to normalize them |
| `molpro_table` | `to_dataframe` / `extract_table`: builds the DataFrame |
| `tabular_output` | `write_table` and a registry of output formats (`register_format` adds new ones). Parquet and Excel need `pyarrow`/`openpyxl` |
