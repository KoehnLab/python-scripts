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

Before anything else, ensure that you have the required dependencies installed on your system. In order to ensure that, simply run
`pip3 install -r requirements.txt` from the root of this repository.

In order to make the provided scripts immediately accessible from your shell, you have to add the `scripts` directory to your `PATH` environment
variable. Assuming you have cloned this repository under `~/Git/python-scripts`, you'll have to add the following entry to your `.bashrc`:
```bash
export PATH="$HOME/Git/python-scripts/scripts/:$PATH"
```
After executing `source ~/.bashrc`, you can execute all scripts inside `scripts` simply by using their name from anywhere on your system. E.g. simply
typing `my_script.py` in the console will execute `scripts/my_script.py` (provided that script exists and there is no other entry inside your `PATH`
that matches this name).


## Package dependencies

Try to limit the dependencies on external Python package to a minimum. Especially when contributing to one of the packages contained in this
repository as that dependency has to be fulfilled by all users of that package, even if they don't use that specific sub-module that has this
dependency.

All dependencies are listed in the `requirements.txt` file and can be installed in one go via `pip3 install -r requirements.txt`. If you add a new
dependency, ensure to add the respective package name to this file.


## Formatting

In order to maintain a consistent code formatting across different scripts and modules throughout this repository, you should use the
[black](https://pypi.org/project/black/) autoformatter on your source code before checking your changes into git.

`black` can be installed via `pip` by `pip install black` (or `pip3 install black`). This will automatically install the `black` executable on your
system with which you can format your code: `black your_file.py`.
Alternatively, you can also format entire directories of source files by giving the path to the respective directory: `black packages/`

In order to bring all source files into the correct format, simply run `black .` from the root of this repository.


## Running tests

In order to run the test cases, make sure that you have set up the `koehnlab` package. Afterwards, simply run `python3 -m unittest` from the
repository's root, which will run all available test cases.

You should always run all test cases before checking in any changes that you made into git. **If the tests fail, don't check your changes in**!
Passing tests should always be an essential requirement for checking in any changes. If there are failing tests, then you broke something with your
changes!

