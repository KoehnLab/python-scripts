"""Parser for the XML output of Molpro.

The parser itself is independent of any table library and of any output format. It turns an XML file into a flat list of
:class:`Record` objects (one per ``<property>`` element) that carry the context in which the value was computed (job step,
basis set, geometry, warnings, ...). Further processing (filtering, naming molecules, building a table, writing files) is
done by other modules.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from lxml import etree

MOLPRO_NS = "http://www.molpro.net/schema/molpro-output"
CML_NS = "http://www.xml-cml.org/schema"


def _q(tag: str, ns: str = MOLPRO_NS) -> str:
    return "{%s}%s" % (ns, tag)


@dataclass
class Record:
    """A single ``<property>`` element together with the context it was reported in"""

    source_path: str
    job_index: int
    # Index of the job step in document order (counting nested steps, e.g. from geometry optimizations, as well)
    step_index: int
    command: str
    commandset: Optional[str]
    # Command of the enclosing job step, if this step is nested (e.g. "OPTG"); None for top-level steps
    parent_command: Optional[str]
    # True if this is the last nested step of its parent
    last_nested: bool
    property: str
    method: Optional[str]
    # Scalar value. None for non-scalar properties (e.g. dipole moments), see raw_value
    value: Optional[float]
    raw_value: str
    principal: bool
    state: Optional[int]
    symmetry: Optional[int]
    multiplicity: Optional[int]
    # Basis set and functional are taken from the <summary> of the step; steps without summary inherit from earlier steps
    basis_set: Optional[str]
    functional: Optional[str]
    # Chemical elements of the geometry that is active in this step (inherited from earlier steps if the step has none)
    elements: tuple = ()
    # "<Type>: <message>" for all <error> elements of the step
    messages: tuple = ()
    errored: bool = False
    # Filled in by name_molecules()
    molecule: Optional[str] = None
    formula: Optional[str] = None


@dataclass
class ParseResult:
    path: str
    records: list = field(default_factory=list)
    # Text of the (original) input for each job, keyed by job index
    inputs: dict = field(default_factory=dict)
    # Human-readable descriptions of anything that went wrong (invalid file, steps with errors, ...)
    problems: list = field(default_factory=list)


def _int_or_none(value: Optional[str]) -> Optional[int]:
    return int(value) if value is not None else None


def _parse_scalar(raw: str) -> Optional[float]:
    try:
        return float(raw.replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _input_text(job) -> Optional[str]:
    input_elem = job.find(_q("input"))
    if input_elem is None:
        return None
    lines = []
    for child in input_elem:
        if child.tag == _q("p"):
            lines.append("".join(child.itertext()).strip())
        elif child.tag == _q("include"):
            # Keep includes as they carry the path to the used geometry file
            lines.append("include %s" % child.get("file", ""))
    return "\n".join(lines)


def _step_elements(job):
    """Yields (step element, parent command or None, is last nested step) in document order"""
    for step in job.iter(_q("jobstep")):
        parent = step.getparent()
        if parent.tag == _q("jobstep"):
            siblings = [s for s in parent if s.tag == _q("jobstep")]
            yield step, parent.get("command"), siblings[-1] is step
        else:
            yield step, None, False


def _elements_of_step(step) -> tuple:
    atoms = step.findall("%s/%s/%s" % (_q("molecule", CML_NS), _q("atomArray", CML_NS), _q("atom", CML_NS)))
    return tuple(a.get("elementType") for a in atoms)


def parse_molpro_xml(path: str) -> ParseResult:
    """Parses the given Molpro XML file. Never raises for malformed content, but reports it in ParseResult.problems."""
    result = ParseResult(path=path)
    try:
        tree = etree.parse(path)
    except (etree.XMLSyntaxError, OSError) as e:
        result.problems.append("Invalid XML: %s" % e)
        return result

    step_index = 0
    for job_index, job in enumerate(tree.getroot().iter(_q("job"))):
        input_text = _input_text(job)
        if input_text is None:
            result.problems.append("Job %d is missing the input element" % job_index)
        else:
            result.inputs[job_index] = input_text

        basis_set = None
        functional = None
        elements: tuple = ()
        for step, parent_command, last_nested in _step_elements(job):
            command = step.get("command", "")

            step_elements = _elements_of_step(step)
            if step_elements:
                elements = step_elements
            summary = step.find(_q("summary"))
            if summary is not None:
                basis_set = summary.get("basis", basis_set)
                functional = summary.get("density_functional")

            messages = tuple(
                "%s: %s" % (e.get("type", "Error"), e.get("message", "")) for e in step.findall(_q("error"))
            )
            errored = any(m.startswith("Error") for m in messages)
            for message in messages:
                result.problems.append("Step %d (%s): %s" % (step_index, command, message))

            for prop in step.findall(_q("property")):
                raw_value = prop.get("value", "")
                result.records.append(
                    Record(
                        source_path=path,
                        job_index=job_index,
                        step_index=step_index,
                        command=command,
                        commandset=step.get("commandset"),
                        parent_command=parent_command,
                        last_nested=last_nested,
                        property=prop.get("name", ""),
                        method=prop.get("method"),
                        value=_parse_scalar(raw_value),
                        raw_value=raw_value,
                        principal=prop.get("principal", "false").lower() == "true",
                        state=_int_or_none(prop.get("stateNumber")),
                        symmetry=_int_or_none(prop.get("stateSymmetry")),
                        multiplicity=_int_or_none(prop.get("stateMultiplicity")),
                        basis_set=basis_set,
                        functional=functional,
                        elements=elements,
                        messages=messages,
                        errored=errored,
                    )
                )
            step_index += 1

    return result


def find_xml_files(search_path: str) -> list:
    """Returns all XML files below the given directory (recursively) or the path itself, if it is a file"""
    if os.path.isfile(search_path):
        return [search_path]
    found = []
    for root, _dirs, files in os.walk(search_path):
        found.extend(os.path.join(root, f) for f in files if f.endswith(".xml"))
    return sorted(found)


def hill_formula(elements) -> str:
    """Molecular formula in Hill order (C first, then H, then alphabetical; purely alphabetical if there is no carbon)"""
    counts: dict = {}
    for element in elements:
        counts[element] = counts.get(element, 0) + 1
    if "C" in counts:
        order = [e for e in ("C", "H") if e in counts] + sorted(e for e in counts if e not in ("C", "H"))
    else:
        order = sorted(counts)
    return "".join(e if counts[e] == 1 else "%s%d" % (e, counts[e]) for e in order)


_GEOMETRY_REGEX = re.compile(r"\bgeom(?:etry)?\s*=\s*([^\s,;{}]+)", re.IGNORECASE)
_RUN_SUFFIX_REGEX = re.compile(r"\.\d+-\d+(?:-[0-9a-f]+)?$")


def xyz_names_from_input(input_text: str) -> list:
    """Names (basename without extension) of all distinct XYZ files referenced by geometry=... in the input, in order"""
    names: list = []
    for match in _GEOMETRY_REGEX.finditer(input_text):
        base = os.path.basename(match.group(1))
        if base.lower().endswith(".xyz") and base[:-4] not in names:
            names.append(base[:-4])
    return names


def name_from_path(path: str) -> str:
    """File name without extension and without the run suffix that Molpro wrappers append (e.g. '.20261119-130642-ab12cd')"""
    stem = os.path.splitext(os.path.basename(path))[0]
    return _RUN_SUFFIX_REGEX.sub("", stem)


def name_molecules(result: ParseResult) -> None:
    """Sets Record.molecule and Record.formula for all records of the parse result (in place).

    The molecule name is, in this order of preference:
    1. the name of the XYZ file given in the input via geometry=<file>.xyz. If the input references several files, they
       are assigned to the distinct formulas in the order in which these occur in the job steps, provided that the
       numbers match. Otherwise this strategy is skipped as it would be ambiguous.
    2. the Hill formula of the geometry (e.g. 'C3H5')
    3. the name of the XML file
    """
    for job_index in sorted({r.job_index for r in result.records}):
        job_records = [r for r in result.records if r.job_index == job_index]
        formulas = [hill_formula(r.elements) if r.elements else None for r in job_records]
        distinct = list(dict.fromkeys(f for f in formulas if f is not None))
        xyz_names = xyz_names_from_input(result.inputs.get(job_index, ""))

        by_formula: dict = {}
        if xyz_names and (len(xyz_names) == len(distinct) or (len(xyz_names) == 1 and not distinct)):
            by_formula = dict(zip(distinct, xyz_names))
        single_xyz = xyz_names[0] if len(xyz_names) == 1 and not distinct else None

        for record, formula in zip(job_records, formulas):
            record.formula = formula
            if formula is not None and formula in by_formula:
                record.molecule = by_formula[formula]
            elif single_xyz is not None:
                record.molecule = single_xyz
            elif formula is not None:
                record.molecule = formula
            else:
                record.molecule = name_from_path(result.path)
