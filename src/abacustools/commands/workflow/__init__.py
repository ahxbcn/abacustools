"""The ``abacustools workflow`` command family."""

from . import bsse as bsse_workflow
from . import bec as bec_workflow
from . import chgdiff as chgdiff_workflow
from . import dftu as dftu_workflow
from . import ecutwfc as ecutwfc_workflow
from . import eos as eos_workflow
from . import exchange as exchange_workflow
from . import elastic as elastic_workflow
from . import fdforce as fdforce_workflow
from . import fdstress as fdstress_workflow
from . import kspacing as kspacing_workflow
from . import phonon as phonon_workflow
from . import piezoelectric as piezoelectric_workflow
from . import thermal_conductivity as thermal_conductivity_workflow
from . import vacancy as vacancy_workflow
from . import vibration as vibration_workflow
from . import workfunc as workfunc_workflow


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
    bec_workflow.register_parser(workflow_subparsers)
    chgdiff_workflow.register_parser(workflow_subparsers)
    dftu_workflow.register_parser(workflow_subparsers)
    ecutwfc_workflow.register_parser(workflow_subparsers)
    eos_workflow.register_parser(workflow_subparsers)
    exchange_workflow.register_parser(workflow_subparsers)
    kspacing_workflow.register_parser(workflow_subparsers)
    elastic_workflow.register_parser(workflow_subparsers)
    fdforce_workflow.register_parser(workflow_subparsers)
    fdstress_workflow.register_parser(workflow_subparsers)
    phonon_workflow.register_parser(workflow_subparsers)
    piezoelectric_workflow.register_parser(workflow_subparsers)
    thermal_conductivity_workflow.register_parser(workflow_subparsers)
    vacancy_workflow.register_parser(workflow_subparsers)
    vibration_workflow.register_parser(workflow_subparsers)
    workfunc_workflow.register_parser(workflow_subparsers)
