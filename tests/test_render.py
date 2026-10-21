"""Tests for the shared rich CLI rendering."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout

from abacustools.core.render import cli_output


def _captured_stdout(call, *args, **kwargs) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        call(*args, **kwargs)
    return buffer.getvalue()


def test_cli_output_matches_print_semantics() -> None:
    cases = [
        (),
        ("plain",),
        (1, "two", 3.5),
        ("a line\nwith breaks\n\n"),
        ("x" * 200,),
        ("brackets stay [literal] here",),
        ({"dict": [1, 2]},),
    ]
    for args in cases:
        assert _captured_stdout(cli_output, *args) == _captured_stdout(print, *args)


def test_cli_output_supports_custom_sep_end_and_streams() -> None:
    assert _captured_stdout(cli_output, "a", "b", sep=", ", end="!\n") == "a, b!\n"

    errors = io.StringIO()
    with redirect_stderr(errors):
        cli_output("warning", stderr=True)
    assert errors.getvalue() == "warning\n"

    buffer = io.StringIO()
    cli_output("routed", file=buffer)
    assert buffer.getvalue() == "routed\n"
