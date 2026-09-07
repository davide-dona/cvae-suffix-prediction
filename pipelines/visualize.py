from collections.abc import Sequence
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from omegaconf import DictConfig

from src.cli import banner, step
from src.evaluation import read_reports
from src.runtime import output_path, start_stage
from src.uncertainty import test_significance
from src.visualization import (
    FIGURES,
    TABLES,
    apply_style,
    compose_figure,
    latex_table,
    reported_models,
)


def _save_figure(figure: Figure, path: Path) -> None:
    """Write one finished figure and close it.

    Args:
        figure: The figure to write, closed afterwards so a run drawing dozens does not hold them
            all open.
        path: Where to write it, from `paths.FIGURE.prepare`.
    """
    figure.savefig(path)
    plt.close(figure)


def _draw_figures(frame: pd.DataFrame) -> int:
    """Draw every figure of the catalogue, each covering every log the reports cover at once.

    Args:
        frame: Every report read, from `read_reports`.
    Returns:
        How many figures were written, under `outputs/visual/figures/`.
    """
    written = 0
    for plot in FIGURES:
        _save_figure(
            figure=compose_figure(frame[frame['axis'].isin(plot.breakdowns)], plot),
            path=output_path(f'figures/{plot.name}.pdf'),
        )
        written += 1
    return written


def _write_tables(frame: pd.DataFrame, significance: pd.DataFrame) -> int:
    """Write every comparison table, over every log at once, under `outputs/visual/tables/`.

    Args:
        frame: Every report read, from `read_reports`.
        significance: Table emphasis and adjusted comparisons from `test_significance`.
    Returns:
        How many tables were written.
    """
    for table in TABLES:
        output_path(f'tables/{table.name}.tex').write_text(latex_table(frame, table, significance))
    return len(TABLES)


def run(evaluation_files: Sequence[Path]) -> None:
    """Draw a set of evaluation reports and tabulate them, under `outputs/visual/`.

    Args:
        evaluation_files: The reports to compare, from `python -m pipelines.evaluate`. These draw
            the metric figures and comparison tables, and the per-prefix scores beside each report
            are what the tables' emphasis is tested on.
    Raises:
        ValueError: If a file is not a report, if a report has no per-prefix scores beside it, if a
            model has no look declared in `src.visualization.labels`, or if one log is given two
            runs of the same model.
    """
    apply_style()
    banner(
        'Drawing the figures and tables',
        {
            'reports': f'{len(evaluation_files)} file(s), with their per-prefix scores beside them',
            'figures': output_path('figures'),
            'tables': output_path('tables'),
        },
    )

    with step(f'Reading {len(evaluation_files)} evaluation report(s)'):
        # Models sharing a style are one model from here on: one line, one column, one legend key.
        reports = reported_models(read_reports(evaluation_files))

    logs = sorted(set(reports['dataset']))
    with step(f'Drawing {", ".join(logs)}'):
        drawn = _draw_figures(reports)

    with step('Comparing means with a paired case bootstrap'):
        significance = reported_models(test_significance(evaluation_files))

    with step('Writing the comparison tables'):
        tables = _write_tables(reports, significance)

    print(
        f'\nWrote {drawn} figures in pdf to {output_path("figures")} '
        f'and {tables} tables in tex to {output_path("tables")}'
    )


@hydra.main(version_base='1.3', config_path='../config', config_name='visualize')
def main(cfg: DictConfig) -> None:
    start_stage(cfg)
    if bool(cfg.evaluations) == bool(cfg.evaluations_dir):
        raise ValueError('Provide evaluations or evaluations_dir, exclusively')
    files = [Path(path) for path in cfg.evaluations]
    if cfg.evaluations_dir:
        files = sorted(
            path for folder in cfg.evaluations_dir for path in Path(folder).rglob('evaluation.json')
        )
    if not files:
        raise ValueError('No evaluation reports found')
    run(files)


if __name__ == '__main__':
    main()
