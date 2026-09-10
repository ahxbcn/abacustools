"""Input/output context for the interactive menu."""

from __future__ import annotations

from typing import Protocol

from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm


class MenuContext(Protocol):
    """Input/output hooks so the menu can run live or under test."""

    def ask(self, prompt: str) -> str: ...

    def confirm(self, prompt: str, default: bool) -> bool: ...

    def write(self, *objects: object) -> None: ...


class ConsoleContext:
    """A :class:`MenuContext` backed by a live rich console."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def ask(self, prompt: str) -> str:
        return self.console.input(f"[bold cyan]{escape(prompt)}[/bold cyan]: ")

    def confirm(self, prompt: str, default: bool) -> bool:
        return Confirm.ask(escape(prompt), default=default, console=self.console)

    def write(self, *objects: object) -> None:
        self.console.print(*objects)


def default_context() -> MenuContext:
    """Build a context backed by a live rich console."""
    return ConsoleContext()
