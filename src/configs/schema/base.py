from pydantic import BaseModel, ConfigDict

# Names become directories: lowercase, digits and hyphens, so a name cannot be empty or escape
# its parent.
NAME_PATTERN = r'^[a-z0-9][a-z0-9-]*$'


class StrictModel(BaseModel):
    """Base for every config section: immutable and typo-proof."""

    model_config = ConfigDict(frozen=True, extra='forbid')
