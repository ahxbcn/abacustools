"""Tests for the ``abacustools mp`` command family."""

from __future__ import annotations

import json
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from abacustools.commands.mp.download import run as run_download
from abacustools.commands.mp.search import run as run_search
from abacustools.integrations.materials_project import (
    MaterialStructure,
    MaterialSummary,
    MaterialsProjectApiKeyError,
)
from abacustools.main import _create_parser


def _structure():
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _summary(material_id: str = "mp-149") -> MaterialSummary:
    return MaterialSummary(
        material_id=material_id,
        formula="Si2",
        chemsys="Si",
        nsites=2,
        volume=40.0,
        energy_above_hull=0.0,
        band_gap=0.61,
        is_stable=True,
        theoretical=False,
    )


def _search_namespace(*arguments: str):
    return _create_parser("abacustools").parse_args(["mp", "search", *arguments])


def _download_namespace(*arguments: str):
    return _create_parser("abacustools").parse_args(["mp", "download", *arguments])


class TestMpParser(unittest.TestCase):
    def test_mp_family_is_registered(self):
        namespace = _search_namespace("--formula", "Si")

        self.assertEqual(namespace.command, "mp")
        self.assertEqual(namespace.mp_command, "search")
        self.assertIsNotNone(namespace.handler)

    def test_download_requires_a_material_id(self):
        parser = _create_parser("abacustools")
        with patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["mp", "download"])

        self.assertEqual(error.exception.code, 2)

    def test_download_rejects_an_unknown_format(self):
        parser = _create_parser("abacustools")
        with patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["mp", "download", "mp-149", "--format", "gen"])

        self.assertEqual(error.exception.code, 2)


class TestMpSearchCommand(unittest.TestCase):
    def test_search_prints_a_table_of_matches(self):
        with patch(
            "abacustools.commands.mp.search.search_materials",
            return_value=[_summary()],
        ):
            namespace = _search_namespace("--formula", "Si", "--limit", "5")
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                status = run_search(namespace)

        self.assertEqual(status, 0)
        self.assertIn("mp-149", stdout.getvalue())
        self.assertIn("Si2", stdout.getvalue())
        self.assertIn("0.610", stdout.getvalue())

    def test_search_passes_the_parsed_filters_to_the_adapter(self):
        with patch(
            "abacustools.commands.mp.search.search_materials",
            return_value=[_summary()],
        ) as search:
            namespace = _search_namespace(
                "--chemsys", "Si-O", "--elements", "Si", "O", "--stable", "--limit", "7"
            )
            with patch("sys.stdout", new_callable=StringIO):
                run_search(namespace)

        self.assertEqual(search.call_args.kwargs["chemsys"], "Si-O")
        self.assertEqual(search.call_args.kwargs["elements"], ["Si", "O"])
        self.assertTrue(search.call_args.kwargs["is_stable"])
        self.assertEqual(search.call_args.kwargs["limit"], 7)

    def test_search_prints_json_and_writes_the_output_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "results.json"
            with patch(
                "abacustools.commands.mp.search.search_materials",
                return_value=[_summary()],
            ):
                namespace = _search_namespace(
                    "--formula", "Si", "--json", "--output", str(destination)
                )
                with patch("sys.stdout", new_callable=StringIO) as stdout:
                    status = run_search(namespace)

            self.assertEqual(status, 0)
            self.assertEqual(json.loads(stdout.getvalue())[0]["material_id"], "mp-149")
            self.assertEqual(json.loads(destination.read_text())[0]["band_gap"], 0.61)

    def test_search_without_matches_returns_one(self):
        with patch("abacustools.commands.mp.search.search_materials", return_value=[]):
            namespace = _search_namespace("--formula", "Xx")
            with patch("sys.stdout", new_callable=StringIO) as stdout:
                status = run_search(namespace)

        self.assertEqual(status, 1)
        self.assertIn("no materials matched", stdout.getvalue())

    def test_search_without_a_selector_fails_before_reaching_the_database(self):
        with patch("sys.stderr", new_callable=StringIO) as stderr:
            status = run_search(_search_namespace())

        self.assertEqual(status, 1)
        self.assertIn("selector", stderr.getvalue())

    def test_search_reports_a_missing_api_key(self):
        with patch(
            "abacustools.commands.mp.search.search_materials",
            side_effect=MaterialsProjectApiKeyError("no Materials Project API key found"),
        ):
            namespace = _search_namespace("--formula", "Si")
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                status = run_search(namespace)

        self.assertEqual(status, 1)
        self.assertIn("no Materials Project API key found", stderr.getvalue())


class TestMpDownloadCommand(unittest.TestCase):
    def test_download_writes_one_directory_per_material(self):
        material = MaterialStructure(summary=_summary(), structure=_structure())

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "abacustools.commands.mp.download.download_material",
                return_value=material,
            ):
                namespace = _download_namespace(
                    "mp-149", "--output", temporary, "--format", "poscar"
                )
                with patch("sys.stdout", new_callable=StringIO) as stdout:
                    status = run_download(namespace)

            written = Path(temporary) / "mp-149" / "POSCAR"
            self.assertTrue(written.is_file())

        self.assertEqual(status, 0)
        self.assertIn("mp-149", stdout.getvalue())
        self.assertIn("POSCAR", stdout.getvalue())

    def test_download_defaults_to_stru(self):
        material = MaterialStructure(summary=_summary(), structure=_structure())

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "abacustools.commands.mp.download.download_material",
                return_value=material,
            ):
                namespace = _download_namespace("mp-149", "-o", temporary)
                with patch("sys.stdout", new_callable=StringIO):
                    status = run_download(namespace)

            written = Path(temporary) / "mp-149" / "STRU"
            self.assertTrue(written.is_file())
            self.assertIn("ATOMIC_SPECIES", written.read_text())

        self.assertEqual(status, 0)

    def test_download_prints_json_when_asked(self):
        material = MaterialStructure(summary=_summary(), structure=_structure())

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "abacustools.commands.mp.download.download_material",
                return_value=material,
            ):
                namespace = _download_namespace("mp-149", "-o", temporary, "--json")
                with patch("sys.stdout", new_callable=StringIO) as stdout:
                    status = run_download(namespace)

        self.assertEqual(status, 0)
        record = json.loads(stdout.getvalue())[0]
        self.assertEqual(record["material_id"], "mp-149")
        self.assertTrue(record["file"].endswith("mp-149/STRU"))

    def test_download_reports_unknown_materials_and_returns_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "abacustools.commands.mp.download.download_material",
                side_effect=LookupError("no Materials Project structure for mp-999"),
            ):
                namespace = _download_namespace("mp-999", "-o", temporary)
                with patch("sys.stderr", new_callable=StringIO) as stderr:
                    status = run_download(namespace)

        self.assertEqual(status, 1)
        self.assertIn("mp-999", stderr.getvalue())

    def test_download_keeps_going_after_one_failure(self):
        material = MaterialStructure(summary=_summary(), structure=_structure())

        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "abacustools.commands.mp.download.download_material",
                side_effect=[LookupError("no Materials Project structure"), material],
            ):
                namespace = _download_namespace("mp-1", "mp-149", "-o", temporary)
                with (
                    patch("sys.stdout", new_callable=StringIO),
                    patch("sys.stderr", new_callable=StringIO),
                ):
                    status = run_download(namespace)

            self.assertTrue((Path(temporary) / "mp-149" / "STRU").is_file())
            self.assertFalse((Path(temporary) / "mp-1").exists())

        self.assertEqual(status, 1)

    def test_download_stops_when_the_client_is_unavailable(self):
        with patch(
            "abacustools.commands.mp.download.download_material",
            side_effect=MaterialsProjectApiKeyError("no key"),
        ):
            namespace = _download_namespace("mp-149")
            with patch("sys.stderr", new_callable=StringIO) as stderr:
                status = run_download(namespace)

        self.assertEqual(status, 1)
        self.assertIn("no key", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
