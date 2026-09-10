"""Interactive navigation over the argparse command tree."""

from __future__ import annotations

import argparse
import shlex
from typing import Optional

from rich.markup import escape
from rich.table import Table

from abacustools.menu.context import MenuContext, default_context
from abacustools.menu.prompt import MenuInputError, prompt_argv, safe_parse
from abacustools.menu.tree import MenuNode, build_menu


class _Quit(Exception):
    pass


class _Back(Exception):
    pass


def _render(node: MenuNode, ctx: MenuContext, prog: str) -> None:
    breadcrumb = " > ".join((prog,) + node.path) if node.path else prog
    table = Table(title=breadcrumb, header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Command", style="bold")
    table.add_column("Description")
    for index, child in enumerate(node.children, start=1):
        table.add_row(str(index), child.title, child.help)
    ctx.write(table)
    ctx.write(escape("[b] back   [q] quit"))


def _select(node: MenuNode, ctx: MenuContext, prog: str) -> MenuNode:
    _render(node, ctx, prog)
    while True:
        raw = ctx.ask("Select").strip()
        if raw in ("q", "quit", "exit"):
            raise _Quit
        if raw in ("b", "back", "0"):
            raise _Back
        if raw.isdigit() and 1 <= int(raw) <= len(node.children):
            return node.children[int(raw) - 1]
        for child in node.children:
            if raw == child.name or raw in child.aliases:
                return child
        ctx.write(escape(f"Invalid selection: {raw!r}"))


def _execute(
    node: MenuNode,
    root_parser: argparse.ArgumentParser,
    ctx: MenuContext,
    prog: str,
) -> None:
    if node.parser is None:
        ctx.write("This menu entry is not runnable.")
        return
    tokens = prompt_argv(node.parser, ctx)
    argv = list(node.path) + list(tokens)
    preview = " ".join(shlex.quote(part) for part in [prog, *argv])
    ctx.write(escape(f"Command: {preview}"))
    try:
        namespace = safe_parse(root_parser, argv)
    except MenuInputError as error:
        ctx.write(escape(f"Invalid input: {error}"))
        return
    handler = getattr(namespace, "handler", None) or node.handler
    if handler is None:
        ctx.write("No handler registered for this command.")
        return
    try:
        code = handler(namespace)
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else 0
    except Exception as error:
        ctx.write(escape(f"Error: {error}"))
        code = 1
    ctx.write(f"Exit code: {code}")
    ctx.ask("Press ENTER to continue")


def run_menu(
    root_parser: argparse.ArgumentParser,
    ctx: Optional[MenuContext] = None,
    prog: str = "abacustools",
) -> int:
    """Run the interactive menu until the user quits."""
    context = ctx or default_context()
    root = build_menu(root_parser)
    if not root.children:
        context.write("No commands available.")
        return 0
    stack = [root]
    try:
        while stack:
            node = stack[-1]
            if node.is_leaf:
                _execute(node, root_parser, context, prog)
                stack.pop()
                continue
            try:
                child = _select(node, context, prog)
            except _Back:
                if len(stack) > 1:
                    stack.pop()
                continue
            stack.append(child)
    except (_Quit, KeyboardInterrupt, EOFError):
        context.write("")
    return 0
