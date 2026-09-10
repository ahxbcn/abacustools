"""Build a navigable menu tree by reflecting the argparse command hierarchy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class MenuNode:
    """One entry in the menu tree: a submenu, or a leaf wrapping a parser."""

    name: str
    path: tuple[str, ...] = ()
    help: str = ""
    aliases: tuple[str, ...] = ()
    parser: Optional[argparse.ArgumentParser] = None
    handler: Optional[Callable[[argparse.Namespace], int]] = None
    children: list["MenuNode"] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.parser is not None

    @property
    def title(self) -> str:
        if self.aliases:
            return f"{self.name} ({', '.join(self.aliases)})"
        return self.name


def _subparsers(parser: argparse.ArgumentParser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    return None


def _choice_help(action: argparse._SubParsersAction) -> dict[str, str]:
    return {
        pseudo.dest: (pseudo.help or "")
        for pseudo in getattr(action, "_choices_actions", [])
    }


def _excluded(parser: argparse.ArgumentParser) -> bool:
    return bool(parser.get_default("_menu_exclude"))


def build_menu(parser: argparse.ArgumentParser, path: tuple[str, ...] = ()) -> MenuNode:
    """Recursively mirror an argparse parser tree as a :class:`MenuNode` tree."""
    node = MenuNode(name=path[-1] if path else "abacustools", path=path)
    subparsers = _subparsers(parser)
    if subparsers is None:
        node.parser = parser
        node.handler = parser.get_default("handler")
        return node

    helps = _choice_help(subparsers)
    seen: dict[int, MenuNode] = {}
    for name, child_parser in subparsers.choices.items():
        if _excluded(child_parser):
            continue
        if id(child_parser) in seen:
            existing = seen[id(child_parser)]
            existing.aliases = existing.aliases + (name,)
            continue
        child = build_menu(child_parser, path + (name,))
        child.help = helps.get(name, "")
        seen[id(child_parser)] = child
        node.children.append(child)
    return node
