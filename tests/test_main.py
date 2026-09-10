"""Tests for the abacustools argparse subcommand parser."""

from __future__ import annotations

import tempfile
import unittest
from unittest.mock import patch

from abacustools.main import _create_parser, main
from abacustools.version import __version__


class TestMain(unittest.TestCase):
    def test_version_subcommand(self) -> None:
        with patch("sys.stdout") as stdout:
            status = main(["version"])

        self.assertEqual(status, 0)
        stdout.write.assert_any_call(f"abacustools {__version__}")

    def test_version_option(self) -> None:
        with patch("sys.stdout") as stdout:
            with self.assertRaises(SystemExit) as error:
                main(["--version"])

        self.assertEqual(error.exception.code, 0)
        stdout.write.assert_any_call(f"abacustools {__version__}\n")

    def test_version_subcommand_rejects_arguments(self) -> None:
        with self.assertRaises(SystemExit) as error:
            main(["version", "extra"])

        self.assertEqual(error.exception.code, 2)

    def test_unknown_command_reports_argparse_error(self) -> None:
        with self.assertRaises(SystemExit) as error:
            main(["missing"])

        self.assertEqual(error.exception.code, 2)

    def test_no_command_prints_help(self) -> None:
        with patch("sys.stdout") as stdout:
            status = main([])

        self.assertEqual(status, 0)
        output = "".join(call.args[0] for call in stdout.write.call_args_list)
        self.assertIn("version", output)


class TestPostprocessAliases(unittest.TestCase):
    def test_post_and_pp_aliases_resolve_to_postprocess(self) -> None:
        parser = _create_parser("abacustools")
        with tempfile.TemporaryDirectory() as directory:
            canonical = parser.parse_args(["postprocess", "result", "-j", directory])
            post = parser.parse_args(["post", "result", "-j", directory])
            short = parser.parse_args(["pp", "result", "-j", directory])

        self.assertIs(post.handler, canonical.handler)
        self.assertIs(short.handler, canonical.handler)


if __name__ == "__main__":
    unittest.main()
