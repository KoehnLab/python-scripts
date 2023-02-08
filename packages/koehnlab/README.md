# package koehnlab

This package is meant to serve as a collection of reusable code-snippets that are used inside our group. The package contains sub-packages categorized
by topic where each package may contain multiple modules (Python files).

**IMPORTANT**: You should not place any Python files in here that are meant to be executed by themselves. If you need to, you have to make sure to not
litter the global namespace. Thus, you'd have to use the following pattern:
```python
def main():
    # Code that should be run, when executing the file directly

if __name__ == "__main__":
    # Call main() function only, if this file was executed directly (instead of only being imported)
    main()
```


## Adding sub-package

In order to add a new sub-package, simply create a new subdirectory and make sure to create a `__init__.py` file in that directory (for its contents
see below). Additionally, you'll have to edit the [__init___.py](__init__.py) file in this directory in order to make sure that the new sub-package is
included in the top-level `koehnlab` package. Suppose we have created a new sub-package called `samplepkg`
```python
from . import samplepkg
```

For stylistic reasons, the new entry should fit in with the existing ones to maintain alphabetic ordering of the different sub-packages.


### Contents

A sub-package may contain as many modules (Python files) as needed. In order to make functions and symbols from those modules available when importing
the respective sub-package (`koehnlab.samplepkg`), the respective symbols must be imported in the sub-package's `__init__.py`.

If all symbols (functions and global variables) from a given module shall be made available, we can make use of a wildcard import. For a module called
`myModule.py`, the import would then look like
```python
from .myModule import *
```
(note the leading dot and the missing file extension)

if you want to only make certain symbols available (e.g. because the module contains some utility functions and/or variables that are not supposed to
be accessed from outside the module itself), proceed as follows:
```python
from .myModule import funcA, funcB, importantVariable
```

