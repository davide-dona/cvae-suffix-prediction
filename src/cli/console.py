import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager


def duration(seconds: float) -> str:
    """Format an elapsed time for a person reading a log.

    Args:
        seconds: How long something took.
    Returns:
        The time as `4.2s`, `3m 12s` or `1h 05m`, whichever fits it.
    """
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, seconds = divmod(int(seconds), 60)
    if minutes < 60:
        return f'{minutes}m {seconds:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'


def banner(title: str, fields: Mapping[str, object]) -> None:
    """Announce what a pipeline is about to do, before the first step of it runs.

    Args:
        title: What the pipeline does, e.g. `Generating suffixes`.
        fields: The settings the run is about to work under, one indented line each, in the order
            given. Names are padded to a common width so the values line up.
    """
    print(title, flush=True)
    width = max((len(name) for name in fields), default=0)
    for name, value in fields.items():
        print(f'  {name:<{width}}  {value}')
    print(flush=True)


@contextmanager
def step(message: str) -> Iterator[None]:
    """Say that a step has started, then how long it took.

    Args:
        message: What the step does, e.g. `Loading the test split`. Printed with an ellipsis
            before the block runs, so a step that takes minutes has already named itself.
    Yields:
        Nothing: the step is the block itself.
    """
    print(f'{message}...', flush=True)
    started = time.perf_counter()
    yield
    print(f'  done in {duration(time.perf_counter() - started)}', flush=True)
