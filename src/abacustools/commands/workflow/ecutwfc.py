"""The ``abacustools workflow ecutwfc`` workflow."""

from __future__ import annotations

from .convergence import ConvergenceSpec, register_workflow


SPEC = ConvergenceSpec(
    name="ecutwfc",
    label="cutoff energy",
    unit="Ry",
    default_output="ecutwfc_convergence.json",
    default_plot="ecutwfc_convergence.png",
    ascending=True,
)


def prepare(args):
    """Prepare cutoff-energy convergence jobs."""
    from .convergence import prepare as _prepare

    return _prepare(args, SPEC)


def postprocess(args):
    """Postprocess cutoff-energy convergence jobs."""
    from .convergence import postprocess as _postprocess

    return _postprocess(args, SPEC)


def register_parser(subparsers) -> None:
    """Register the cutoff-energy convergence workflow."""
    register_workflow(subparsers, SPEC, aliases=["cutoff"])
