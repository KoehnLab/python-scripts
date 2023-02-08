# Test cases

This directory contains (unit) tests for the different components provided by this repository. Ideally, all exported functions inside the `koehnlab`
package should have at least a single test-case that checks that the function works as expected.

We are using the [unittest](https://docs.python.org/3/library/unittest.html) framework that ships with Python by default (no additional installation
required).

Please see the existing test cases and the respective `unittest` documentation for how test cases can be written.


## File naming

In order to make tests automatically discoverable by a call to `python3 -m unittest` from the repository's root, all files implementing test cases
must start with `test_` and must only contain letters, digits and underscores. In particular whitespace and dashes ("-") are not allowed due to the
way automatically running all test cases works.

