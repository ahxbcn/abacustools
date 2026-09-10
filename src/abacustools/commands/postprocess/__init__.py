"""The ``abacustools postprocess`` command family."""

from . import bader as bader_command
from . import band as band_command
from . import cohp as cohp_command
from . import dos as dos_command
from . import mayer as mayer_command
from . import result as result_command


def register_parser(subparsers) -> None:
    """Register ``postprocess`` and its nested subcommands."""
    postprocess_parser = subparsers.add_parser(
        "postprocess",
        aliases=["post", "pp"],
        help="Postprocess ABACUS calculation results.",
    )
    postprocess_subparsers = postprocess_parser.add_subparsers(
        dest="postprocess_command",
        metavar="POSTPROCESS_COMMAND",
        title="postprocess subcommands",
        required=True,
    )
    bader_command.register_parser(postprocess_subparsers)
    band_command.register_parser(postprocess_subparsers)
    cohp_command.register_parser(postprocess_subparsers)
    dos_command.register_parser(postprocess_subparsers)
    mayer_command.register_parser(postprocess_subparsers)
    result_command.register_parser(postprocess_subparsers)
