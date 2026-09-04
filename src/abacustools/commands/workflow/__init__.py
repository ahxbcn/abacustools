"""The ``abacustools workflow`` command family."""

from . import bsse as bsse_workflow
from . import chgdiff as chgdiff_workflow
from . import elastic as elastic_workflow
from . import phonon as phonon_workflow


def register_parser(subparsers) -> None:
    """Register workflow tasks and their preparation/postprocessing stages."""
    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Run multi-step ABACUS workflows.",
    )
    workflow_subparsers = workflow_parser.add_subparsers(
        dest="workflow_command",
        metavar="WORKFLOW",
        title="workflow commands",
        required=True,
    )
    bsse_workflow.register_parser(workflow_subparsers)
    chgdiff_workflow.register_parser(workflow_subparsers)
    elastic_workflow.register_parser(workflow_subparsers)
    phonon_workflow.register_parser(workflow_subparsers)
