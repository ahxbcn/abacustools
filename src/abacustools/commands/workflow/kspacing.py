"""The ``abacustools workflow kspacing`` workflow."""

from __future__ import annotations

from .convergence import ConvergenceSpec, register_workflow


SPEC = ConvergenceSpec(
    name="kspacing",
    label="k-point spacing",
    unit="1/Angstrom",
    default_output="kspacing_convergence.json",
    default_plot="kspacing_convergence.png",
    ascending=False,
    requires_kpt=False,
    disable_gamma_only=True,
)


def prepare(args):
    """Prepare k-point-spacing convergence jobs."""
    from .convergence import prepare as _prepare

    return _prepare(args, SPEC)


def postprocess(args):
    """Postprocess k-point-spacing convergence jobs."""
    from .convergence import postprocess as _postprocess

    return _postprocess(args, SPEC)


def register_parser(subparsers) -> None:
    """Register the k-point-spacing convergence workflow."""
    register_workflow(subparsers, SPEC)
