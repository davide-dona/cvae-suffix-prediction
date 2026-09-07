from dataclasses import dataclass

from src.scalar_metrics import Direction, Metric


@dataclass(frozen=True)
class MetricEntry:
    """A metric's label and display settings in a table or figure."""

    metric: Metric
    label: str

    @property
    def key(self) -> str:
        """Return the metric key used in report rows.

        Returns:
            Metric key.
        """
        return self.metric.key

    @property
    def bounds(self) -> tuple[float | None, float | None]:
        """Return optional y-axis bounds for the metric.

        Returns:
            Lower and upper bounds, if declared.
        """
        return self.metric.unit.bounds

    @property
    def shares_scale(self) -> bool:
        """Return whether the metric has fixed [0, 1] bounds.

        Returns:
            Whether all panels share a fixed scale.
        """
        return self.bounds == (0.0, 1.0)

    @property
    def _unit_suffix(self) -> str:
        """Return the bracketed unit suffix, if any.

        Returns:
            Unit suffix or an empty string.
        """
        symbol = self.metric.unit.symbol
        return f' [{symbol}]' if symbol else ''

    @property
    def axis_label(self) -> str:
        """Return an axis label with unit and optimization direction.

        Returns:
            Formatted axis label.
        """
        return f'{self.label}{self._unit_suffix}{self._arrow}'

    @property
    def table_header(self) -> str:
        """Return a unitless LaTex table header with an optimization marker.

        Returns:
            Formatted LaTex header.
        """
        return f'{self.label}{_TABLE_ARROWS[self.metric.direction]}'

    @property
    def _arrow(self) -> str:
        """Return the optimization marker for an axis label.

        Returns:
            Direction marker or an empty string.
        """
        return _AXIS_ARROWS[self.metric.direction]

    def format(self, value: float) -> str:
        """Format a table value to three decimals.

        Args:
            value: Metric value.

        Returns:
            Decimal-formatted value.
        """
        return f'{value:.3f}'


# Direction markers for Matplotlib and LaTex labels.
_AXIS_ARROWS = {
    Direction.HIGHER: '\N{NO-BREAK SPACE}↑',
    Direction.LOWER: '\N{NO-BREAK SPACE}↓',
    Direction.ZERO: '\N{NO-BREAK SPACE}→0',
    Direction.NONE: '',
}
_TABLE_ARROWS = {
    Direction.HIGHER: r'~$\uparrow$',
    Direction.LOWER: r'~$\downarrow$',
    Direction.ZERO: r'~$\rightarrow 0$',
    Direction.NONE: '',
}
