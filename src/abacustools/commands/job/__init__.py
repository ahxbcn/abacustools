"""The ``abacustools job`` command family."""

from . import monitor as monitor_command
from . import prepare as prepare_command
from . import status as status_command
from . import validate as validate_command


def register_parser(subparsers) -> None:
    """Register commands that operate on one ABACUS job directory."""
    job_parser = subparsers.add_parser(
        "job",
        help="Prepare and inspect one ABACUS calculation directory.",
    )
    job_subparsers = job_parser.add_subparsers(
        dest="job_command",
        metavar="JOB_COMMAND",
        title="job commands",
        required=True,
    )
    prepare_command.register_parser(job_subparsers)
    validate_command.register_parser(job_subparsers)
    status_command.register_parser(job_subparsers)
    monitor_command.register_parser(job_subparsers)
