from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from src.logs.declare.templates import TEMPLATES, Constraint

if TYPE_CHECKING:
    from src.configs import DeclareConfig

# Parse the declarative model written by `discover_declare_model` back into the constraints.
_CONSTRAINT_LINE = re.compile(r'^(.*)\[(.*)\]\s*(.*)$')
_TEMPLATE_AND_CARDINALITY = re.compile(r'(^.+?)(\d*$)')
COMMENT = '#'
SETTINGS_LINE = '# settings: '


def discovery_settings(path: Path) -> DeclareConfig | None:
    """Read the header of a declarative model and return the settings it records about how it
    was mined.
    Args:
        path: The model file, from `paths.DECLARE_MODEL`.
    Returns:
        The settings its header records, or `None` for a model written before the header existed,
        which says nothing about how it was mined.
    Raises:
        pydantic.ValidationError: If the header is there but does not describe a discovery.
    """
    from src.configs import DeclareConfig

    for line in path.read_text().splitlines():
        if line.startswith(SETTINGS_LINE):
            return DeclareConfig.model_validate_json(line.removeprefix(SETTINGS_LINE))
    return None


def read_constraints(path: Path) -> list[Constraint]:
    """Read a written declarative model and return the constraints it holds.

    Args:
        path: The model file produced by `discover_declare_model`.
    Returns:
        One entry per constraint, in the order the file holds them.
    Raises:
        ValueError: If a line names a template `TEMPLATES` does not hold, if a binary constraint
            does not name two activities, or if it names the same one twice. Each would silently
            change every conformance number in a report, so none is skipped.
    """
    constraints = []

    for raw in path.read_text().splitlines():
        line = raw.strip()

        # Skip the header, which says how the model was mined rather than what it holds, and any
        # other line that is not a constraint.
        if line.startswith(COMMENT) or not _CONSTRAINT_LINE.search(line):
            continue

        head, rest = line.split('[', 1)
        named = _TEMPLATE_AND_CARDINALITY.search(head)
        if named is None:
            raise ValueError(f'"{line}" does not name a template.')

        name, cardinality = named.group(1), named.group(2)
        template = TEMPLATES.get(name)
        if template is None:
            raise ValueError(
                f'"{line}" uses the {name} template, which src.logs.declare.templates does '
                f'not check. Add it to TEMPLATES there, or mine the model without it.'
            )

        activities = rest.split(']')[0].split(', ')
        expected = 2 if template.is_binary else 1
        if len(activities) != expected:
            raise ValueError(f'"{line}" names {len(activities)} activities, not {expected}.')
        if template.is_binary and activities[0] == activities[1]:
            raise ValueError(
                f'"{line}" names one activity twice, which no template here is defined for.'
            )

        constraints.append(
            Constraint(
                template=template,
                first=activities[0],
                second=activities[1] if template.is_binary else None,
                n=int(cardinality) if template.supports_cardinality and cardinality else 1,
            )
        )
    return constraints
