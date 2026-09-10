"""Collect command-line tokens for a leaf parser through interactive prompts."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from typing import Optional, Sequence

from rich.markup import escape

from abacustools.menu.context import MenuContext


class MenuInputError(Exception):
    """Raised when collected input fails argparse validation."""


@contextmanager
def _no_exit(parser: argparse.ArgumentParser):
    original_error = parser.error
    original_exit = parser.exit

    def _error(message):
        raise MenuInputError(str(message))

    def _exit(status=0, message=None):
        raise MenuInputError(str(message if message is not None else ""))

    parser.error = _error
    parser.exit = _exit
    try:
        yield
    finally:
        parser.error = original_error
        parser.exit = original_exit


def safe_parse(parser: argparse.ArgumentParser, argv: Sequence[str]) -> argparse.Namespace:
    """Parse ``argv`` while preventing argparse from calling ``sys.exit``."""
    with _no_exit(parser):
        return parser.parse_args(list(argv))


_FLAG_ACTIONS = (
    argparse._StoreTrueAction,
    argparse._StoreFalseAction,
    argparse._StoreConstAction,
)

_SKIP_ACTIONS = (
    argparse._HelpAction,
    argparse._VersionAction,
    argparse._SubParsersAction,
)


def _is_promptable(action: argparse.Action) -> bool:
    return action.dest != argparse.SUPPRESS and not isinstance(action, _SKIP_ACTIONS)


def _label(action: argparse.Action) -> str:
    names = ", ".join(action.option_strings) if action.option_strings else action.dest
    return f"{names} - {action.help}" if action.help else names


def _option(action: argparse.Action) -> Optional[str]:
    return action.option_strings[0] if action.option_strings else None


def _resolve_choice(raw: str, choices: Sequence) -> Optional[str]:
    if raw.isdigit():
        index = int(raw) - 1
        if 0 <= index < len(choices):
            return str(choices[index])
    for choice in choices:
        if str(choice) == raw:
            return str(choice)
    return None


def _ask(action: argparse.Action, ctx: MenuContext, prompt: Optional[str] = None) -> str:
    suffix = ""
    if action.choices:
        suffix += " {" + ", ".join(str(choice) for choice in action.choices) + "}"
    if action.default is not None and not action.required:
        suffix += f" [default: {action.default}]"
    return ctx.ask(f"{prompt or _label(action)}{suffix}").strip()


def _prompt_flag(action: argparse.Action, ctx: MenuContext) -> list[str]:
    if ctx.confirm(_label(action) + "?", bool(action.default)):
        return [action.option_strings[0]]
    return []


def _prompt_choice(action: argparse.Action, ctx: MenuContext, multiple: bool) -> list[str]:
    choices = list(action.choices or [])
    for index, choice in enumerate(choices, start=1):
        ctx.write(escape(f"  {index}) {choice}"))
    if multiple:
        raw = _ask(action, ctx, prompt=f"{_label(action)} (numbers/values, blank to finish)")
        if not raw:
            return []
        values = []
        for part in raw.replace(",", " ").split():
            value = _resolve_choice(part, choices)
            if value is None:
                ctx.write(escape(f"Unknown choice: {part}"))
                return []
            values.append(value)
        return values
    raw = _ask(action, ctx, prompt=f"{_label(action)} (number or value)")
    if not raw:
        return []
    value = _resolve_choice(raw, choices)
    if value is None:
        ctx.write(escape(f"Unknown choice: {raw}"))
        return []
    return [value]


def _collect_until_blank(action: argparse.Action, ctx: MenuContext) -> list[str]:
    values: list[str] = []
    while True:
        raw = _ask(action, ctx, prompt=f"{_label(action)} (blank to finish)")
        if not raw:
            break
        values.append(raw)
    return values


def _collect_count(action: argparse.Action, ctx: MenuContext, count: int) -> list[str]:
    values: list[str] = []
    while len(values) < count:
        raw = _ask(action, ctx, prompt=f"{_label(action)} ({len(values) + 1}/{count})")
        if not raw:
            return []
        values.extend(raw.split())
    return values[:count]


def _prompt_append(action: argparse.Action, ctx: MenuContext) -> list[str]:
    option = _option(action)
    nargs = action.nargs
    tokens: list[str] = []
    while True:
        if isinstance(nargs, int):
            values: list[str] = []
            while len(values) < nargs:
                raw = _ask(
                    action,
                    ctx,
                    prompt=f"{_label(action)} ({len(values) + 1}/{nargs}, blank to skip)",
                )
                if not raw:
                    break
                values.extend(raw.split())
            if len(values) < nargs:
                break
        else:
            values = _collect_until_blank(action, ctx)
            if not values:
                break
        if option:
            tokens.append(option)
        tokens.extend(values)
        if not ctx.confirm("Add another entry?", False):
            break
    return tokens


def _prompt_value(action: argparse.Action, ctx: MenuContext) -> list[str]:
    option = _option(action)
    nargs = action.nargs
    prefix = [option] if option else []

    if isinstance(action, argparse._AppendAction):
        return _prompt_append(action, ctx)

    if action.choices:
        multiple = nargs in ("+", "*") or isinstance(action, argparse._ExtendAction)
        values = _prompt_choice(action, ctx, multiple)
        return prefix + values if values else []

    if isinstance(action, argparse._ExtendAction) or nargs in ("+", "*"):
        values = _collect_until_blank(action, ctx)
        return prefix + values if values else []

    if isinstance(nargs, int):
        values = _collect_count(action, ctx, nargs)
        return prefix + values if values else []

    while True:
        raw = _ask(action, ctx)
        if raw:
            return prefix + [raw]
        if not action.required:
            return []
        ctx.write("This value is required.")


def prompt_argv(parser: argparse.ArgumentParser, ctx: MenuContext) -> list[str]:
    """Prompt for every argument of ``parser`` and return the resulting tokens."""
    tokens: list[str] = []
    answered: set[int] = set()
    groups: dict[int, object] = {}
    for group in parser._mutually_exclusive_groups:
        for action in group._group_actions:
            groups[id(action)] = group

    for action in parser._actions:
        if not _is_promptable(action):
            continue
        group = groups.get(id(action))
        if group is not None and id(group) in answered:
            continue
        if isinstance(action, _FLAG_ACTIONS):
            result = _prompt_flag(action, ctx)
        else:
            result = _prompt_value(action, ctx)
        if group is not None and result:
            answered.add(id(group))
        tokens.extend(result)
    return tokens
