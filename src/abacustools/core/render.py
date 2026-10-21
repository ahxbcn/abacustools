"""Shared rich rendering for CLI output.

The CLI layer renders through rich.  The plain informative prints of the
commands go through :func:`cli_output`, which prints text with the exact
semantics of :func:`print` (same byte output on a terminal) while routing
it through a rich ``Console`` so that styling can be adopted per call site.
"""

from __future__ import annotations

from typing import IO, Optional

from rich.console import Console


def cli_output(
    *values: object,
    sep: str = " ",
    end: str = "\n",
    stderr: bool = False,
    file: Optional[IO[str]] = None,
) -> None:
    """Print CLI output through rich with friendly print semantics.

    Args:
        values: The objects to print; rendered with ``str()`` like ``print``.
        sep: Separator inserted between the values.
        end: Text written after the values, ``"\\n"`` by default.
        stderr: Write to stderr instead of stdout (ignored with ``file``).
        file: Optional file object that overrides the default stream.

    The console parses no markup, applies no highlighting and soft-wraps
    nothing, so the printed text matches what :func:`print` produces.
    """
    kwargs: dict = {"markup": False, "highlight": False, "soft_wrap": True}
    if file is not None:
        kwargs["file"] = file
    else:
        kwargs["stderr"] = stderr
    console = Console(**kwargs)
    text = sep.join(str(value) for value in values)
    console.print(text, end=end)
