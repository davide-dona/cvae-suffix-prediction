from pydantic import Field

from .base import StrictModel


class InferenceConfig(StrictModel):
    """How many suffixes a prefix is answered with on each of the two generation passes, and the
    decoder memory bound the two share.

    The two counts are set to the same number: EMSC read at a smaller budget is a different
    number rather than a noisier one, so a checkpoint selected at a smaller one is selected on a
    score the report never reports. They stay separate fields because they answer for different
    populations - a fixed slice of the validation split against the whole test split - and because
    `pipelines.generate`'s `-n/--num-samples` overrides the evaluation one alone.
    """

    validation_samples: int = Field(
        ..., ge=10, description='Draws per prefix in training; minimum 10 for `hit_rate_at_10`'
    )
    evaluation_samples: int = Field(
        ...,
        ge=10,
        description='Draws per prefix in `pipelines.generate`; minimum 10 for `hit_rate_at_10`',
    )
    generation_rows_upper_bound: int = Field(
        ...,
        gt=0,
        description='Upper bound on rows put through the decoder per call, where one prefix is as '
        'many rows as its pass draws samples. Bounds decoder memory independently of that sample '
        'count and of the DataLoader batch size',
    )
