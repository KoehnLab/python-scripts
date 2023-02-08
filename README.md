# python-scripts

Collection of various Python scripts and utilities. Please read through this README before making changes to the repository.

## Organization

This repository is structured in the following way:
- `packages`: This directory contains top-level packages provided by this repository. Most notably this contains the `koehnlab` package and everything
  you want to add to that package needs to go into `packages/koehnlab`. See the contained [README](packages/koehnlab/README.md) for further
  instructions.
- `scripts`: This directory contains Python scripts that can be executed themselves to perform specific tasks. Some of these scripts may depend on the
  `koehnlab` package, so make sure you have set it up correctly.
- `tests`: This directory contains (unit) tests for the provided functionality. You are encouraged to write test cases for your code to ensure that it
  works correctly and future changes won't break things. See the respective [README](tests/README.md) for more information on that.


## The `koehnlab` package

This repository contains reusable Python packages that are bundled together in the top-level `koehnlab` package. In order to make use of this package,
you'll have to set your `PYTHONPATH` environment variable to include the path to the `packages` directory in this repository.

Assuming you have cloned this repository under `~/Git/python-scripts`, then you'll have to add the following entry to your `.bashrc`:
```bash
export PYTHONPATH="$HOME/Git/python-scripts/packages"
```
after you have sourced the edited `.bashrc` (`source ~/.bashrc`), you can simply use the `koehnlab` package. E.g.

```python
from koehnlab import finite_differences

print(finite_differences.forward_difference([1,2], 0.5))
```


## Making scripts available

In order to make the provided scripts immediately accessible from your shell, you have to add the `scripts` directory to your `PATH` environment
variable. Assuming you have cloned this repository under `~/Git/python-scripts`, you'll have to add the following entry to your `.bashrc`:
```bash
export PATH="$HOME/Git/python-scripts/scripts/:$PATH"
```
After executing `source ~/.bashrc`, you can execute all scripts inside `scripts` simply by using their name from anywhere on your system. E.g. simply
typing `my_script.py` in the console will execute `scripts/my_script.py` (provided that script exists and there is no other entry inside your `PATH`
that matches this name).


## Formatting

In order to maintain a consistent code formatting across different scripts and modules throughout this repository, you should use the
[black](https://pypi.org/project/black/) autoformatter on your source code before checking your changes into git.

`black` can be installed via `pip` by `pip install black` (or `pip3 install black`). This will automatically install the `black` executable on your
system with which you can format your code: `black your_file.py`.
Alternatively, you can also format entire directories of source files by giving the path to the respective directory: `black packages/`

In order to bring all source files into the correct format, simply run `black .` from the root of this repository.

