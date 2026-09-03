"""The ``abacustools postprocess`` command family."""

from . import result as result_command


def register_parser(subparsers) -> None:
    """Register ``postprocess`` and its nested subcommands."""
    postprocess_parser = subparsers.add_parser(
        "postprocess",
        help="Postprocess ABACUS calculation results.",
    )
    postprocess_subparsers = postprocess_parser.add_subparsers(
        dest="postprocess_command",
        metavar="POSTPROCESS_COMMAND",
        title="postprocess subcommands",
        required=True,
    )
    result_command.register_parser(postprocess_subparsers)
