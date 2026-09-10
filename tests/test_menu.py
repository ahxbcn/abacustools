"""Tests for the interactive menu engine."""

from __future__ import annotations

import argparse
import unittest

from abacustools.main import _create_parser
from abacustools.menu.prompt import MenuInputError, prompt_argv, safe_parse
from abacustools.menu.runner import run_menu
from abacustools.menu.tree import build_menu


class FakeContext:
    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.messages: list[tuple[object, ...]] = []

    def ask(self, prompt: str) -> str:
        if not self._answers:
            raise EOFError
        return self._answers.pop(0)

    def confirm(self, prompt: str, default: bool) -> bool:
        if not self._answers:
            return default
        raw = self._answers.pop(0).strip().lower()
        if raw == "":
            return default
        return raw in ("y", "yes")

    def write(self, *objects: object) -> None:
        self.messages.append(objects)


def _child(node, name):
    return next(child for child in node.children if child.name == name)


class TestBuildMenu(unittest.TestCase):
    def test_mirrors_families_and_excludes_menu(self) -> None:
        root = build_menu(_create_parser("abacustools"))
        self.assertEqual(
            [child.name for child in root.children],
            ["file", "job", "postprocess", "workflow"],
        )

    def test_captures_family_aliases(self) -> None:
        root = build_menu(_create_parser("abacustools"))
        postprocess = _child(root, "postprocess")
        self.assertIn("post", postprocess.aliases)
        self.assertIn("pp", postprocess.aliases)

    def test_nested_workflow_stages_are_leaves(self) -> None:
        root = build_menu(_create_parser("abacustools"))
        phonon = _child(_child(root, "workflow"), "phonon")
        self.assertEqual([child.name for child in phonon.children], ["prepare", "postprocess"])
        self.assertTrue(all(child.is_leaf for child in phonon.children))

    def test_leaves_expose_parser_and_handler(self) -> None:
        root = build_menu(_create_parser("abacustools"))
        bader = _child(_child(root, "postprocess"), "bader")
        self.assertTrue(bader.is_leaf)
        self.assertIsNotNone(bader.handler)
        self.assertIsNotNone(bader.parser)
        assert bader.parser is not None
        dests = {action.dest for action in bader.parser._actions}
        self.assertIn("job", dests)
        self.assertIn("json", dests)


class TestPrompt(unittest.TestCase):
    def test_required_positional_and_declined_flag(self) -> None:
        parser = argparse.ArgumentParser(prog="demo")
        parser.add_argument("name")
        parser.add_argument("--loud", action="store_true")
        parser.set_defaults(handler=lambda ns: 0)
        self.assertEqual(prompt_argv(parser, FakeContext(["Alice", ""])), ["Alice"])

    def test_accepted_flag_adds_option(self) -> None:
        parser = argparse.ArgumentParser(prog="demo")
        parser.add_argument("--loud", action="store_true")
        parser.set_defaults(handler=lambda ns: 0)
        self.assertEqual(prompt_argv(parser, FakeContext(["y"])), ["--loud"])

    def test_choice_selected_by_number(self) -> None:
        parser = argparse.ArgumentParser(prog="demo")
        parser.add_argument("--mode", choices=["a", "b", "c"])
        parser.set_defaults(handler=lambda ns: 0)
        self.assertEqual(prompt_argv(parser, FakeContext(["2"])), ["--mode", "b"])

    def test_mutually_exclusive_prompts_only_one(self) -> None:
        parser = argparse.ArgumentParser(prog="demo")
        group = parser.add_mutually_exclusive_group()
        group.add_argument("--a", action="store_true")
        group.add_argument("--b", action="store_true")
        parser.set_defaults(handler=lambda ns: 0)
        self.assertEqual(prompt_argv(parser, FakeContext(["y"])), ["--a"])

    def test_variadic_option_collects_until_blank(self) -> None:
        parser = argparse.ArgumentParser(prog="demo")
        parser.add_argument("--tags", nargs="+")
        parser.set_defaults(handler=lambda ns: 0)
        tokens = prompt_argv(parser, FakeContext(["x", "y", ""]))
        self.assertEqual(tokens, ["--tags", "x", "y"])

    def test_safe_parse_reports_error_without_exiting(self) -> None:
        parser = argparse.ArgumentParser(prog="demo")
        parser.add_argument("--count", type=int, required=True)
        with self.assertRaises(MenuInputError):
            safe_parse(parser, ["--count", "abc"])

    def test_safe_parse_returns_namespace(self) -> None:
        parser = argparse.ArgumentParser(prog="demo")
        parser.add_argument("--count", type=int, required=True)
        self.assertEqual(safe_parse(parser, ["--count", "3"]).count, 3)


class TestRunMenu(unittest.TestCase):
    def _parser(self, calls):
        parser = argparse.ArgumentParser(prog="demo")
        parser.set_defaults(_prog="demo")
        subparsers = parser.add_subparsers(dest="command")
        greet = subparsers.add_parser("greet", help="Greet someone")
        greet.add_argument("name")
        greet.add_argument("--loud", action="store_true")
        greet.set_defaults(handler=lambda ns: calls.append((ns.name, ns.loud)) or 0)
        return parser

    def test_navigates_and_executes_selected_command(self) -> None:
        calls = []
        ctx = FakeContext(["1", "Alice", "", "", "q"])
        status = run_menu(self._parser(calls), ctx=ctx, prog="demo")
        self.assertEqual(status, 0)
        self.assertEqual(calls, [("Alice", False)])

    def test_quit_from_root(self) -> None:
        calls = []
        status = run_menu(self._parser(calls), ctx=FakeContext(["q"]), prog="demo")
        self.assertEqual(status, 0)
        self.assertEqual(calls, [])

    def test_back_returns_to_parent(self) -> None:
        calls = []
        ctx = FakeContext(["b", "q"])
        status = run_menu(self._parser(calls), ctx=ctx, prog="demo")
        self.assertEqual(status, 0)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
