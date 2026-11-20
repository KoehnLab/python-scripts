"""Selection of results from parsed Molpro XML files.

This module sits between the parser (:mod:`koehnlab.io.molpro_xml`) and the output stage. It decides which of the parsed
properties are of interest and normalizes them. The selection is controlled by an :class:`ExtractionSpec`.
"""

from dataclasses import dataclass, replace
from typing import Iterable, Optional

from .molpro_xml import ParseResult, Record

NESTED_STEP_MODES = ("none", "last", "all")

DISPERSION_PROPERTY = "dispersion correction"
# Properties that the dispersion correction is attached to the method label of
_DISPERSION_TARGETS = ("energy", "total energy")


@dataclass
class ExtractionSpec:
    """Describes which results to extract"""

    # Names of the properties to extract (case-insensitive). None means all properties.
    properties: Optional[tuple] = ("Energy", "Total energy", "Exc. energy")
    # Only consider properties that Molpro flags as principal
    principal_only: bool = True
    # Which steps nested in another step (e.g. the single point calculations of a geometry optimization) to consider:
    # "none", "last" (only the last one of each parent step) or "all"
    nested_steps: str = "last"
    # Ignore all steps for which Molpro reported an error
    skip_errored_steps: bool = True
    # Append the label of a reported dispersion correction to the method of energies, e.g. "RKS+D4"
    merge_dispersion: bool = True
    # Turn commands like "DF-RHF-SCF" into "DF-RHF" unless the input explicitly asks for the longer name
    strip_scf_suffix: bool = True

    def __post_init__(self):
        if self.nested_steps not in NESTED_STEP_MODES:
            raise ValueError("nested_steps must be one of %s, got '%s'" % (", ".join(NESTED_STEP_MODES), self.nested_steps))


def _wanted_step(record: Record, spec: ExtractionSpec) -> bool:
    if record.parent_command is None:
        return True
    return spec.nested_steps == "all" or (spec.nested_steps == "last" and record.last_nested)


def _dispersion_labels(records: Iterable[Record]) -> dict:
    """Maps (job_index, step_index) to the label of the dispersion correction (e.g. 'D4' from 'PBE D4') of that step"""
    labels: dict = {}
    for record in records:
        if record.property.lower() == DISPERSION_PROPERTY and record.method and " " in record.method.strip():
            labels[(record.job_index, record.step_index)] = record.method.split()[-1]
    return labels


def select_records(result: ParseResult, spec: ExtractionSpec = ExtractionSpec()) -> list:
    """Returns the records of the parse result that match the spec, with the normalizations of the spec applied.

    The records of the parse result are not modified.
    """
    wanted = None if spec.properties is None else {p.lower() for p in spec.properties}
    dispersion = _dispersion_labels(result.records) if spec.merge_dispersion else {}

    selected = []
    for record in result.records:
        if not _wanted_step(record, spec):
            continue
        if spec.skip_errored_steps and record.errored:
            continue
        if spec.principal_only and not record.principal:
            continue
        name = record.property.lower()
        if wanted is not None and name not in wanted:
            continue

        changes: dict = {}
        label = dispersion.get((record.job_index, record.step_index))
        if label is not None and name in _DISPERSION_TARGETS and record.method:
            changes["method"] = "%s+%s" % (record.method, label)
        command = record.command
        if spec.strip_scf_suffix and command.upper().endswith("-SCF"):
            if command.lower() not in result.inputs.get(record.job_index, "").lower():
                changes["command"] = command[: -len("-SCF")]
        selected.append(replace(record, **changes) if changes else record)
    return selected
