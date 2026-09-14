"""Tests for the optional Materials Project database adapters."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from abacustools.integrations.materials_project import (
    DEFAULT_SUMMARY_FIELDS,
    MaterialSummary,
    MaterialsProjectApiKeyError,
    MaterialsProjectUnavailableError,
    api_key_from_environment,
    download_material,
    load_materials_project,
    material_directory,
    materials_project_available,
    resolve_api_key,
    search_materials,
    write_material_structure,
)

_UNSET = object()


def _structure():
    from pymatgen.core import Lattice, Structure

    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )


def _document(**overrides):
    document = {
        "material_id": "mp-149",
        "formula_pretty": "Si2",
        "chemsys": "Si",
        "nsites": 2,
        "volume": 40.0,
        "energy_above_hull": 0.0,
        "band_gap": 0.61,
        "is_stable": True,
        "theoretical": False,
    }
    document.update(overrides)
    return document


class FakeClient:
    """Minimal stand-in for ``MPRester`` that never touches the network."""

    def __init__(self, documents=None, structure=_UNSET):
        self.documents = [_document()] if documents is None else list(documents)
        self.structure = _structure() if structure is _UNSET else structure
        self.search_kwargs = None
        self.requested_ids: list[str] = []
        self.materials = self

    @property
    def summary(self):
        return self

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return list(self.documents)

    def get_structure_by_material_id(self, material_id):
        self.requested_ids.append(material_id)
        return self.structure


class TestMaterialsProjectAvailability(unittest.TestCase):
    def test_missing_optional_dependency_has_actionable_error(self):
        with patch.dict(sys.modules, {"mp_api": None, "mp_api.client": None}):
            with self.assertRaises(MaterialsProjectUnavailableError) as error:
                load_materials_project()

        self.assertIn("mp-api", str(error.exception))

    def test_availability_follows_the_client_import(self):
        with patch.dict(sys.modules, {"mp_api": None, "mp_api.client": None}):
            self.assertFalse(materials_project_available())


class TestMaterialsProjectApiKey(unittest.TestCase):
    def test_explicit_key_wins_over_the_environment(self):
        self.assertEqual(
            resolve_api_key("explicit", {"MP_API_KEY": "from-environment"}),
            "explicit",
        )

    def test_environment_keys_are_used_in_order(self):
        self.assertEqual(api_key_from_environment({"MP_API_KEY": "first"}), "first")
        self.assertEqual(api_key_from_environment({"PMG_MAPI_KEY": "legacy"}), "legacy")
        self.assertEqual(api_key_from_environment({"MAPI_KEY": "oldest"}), "oldest")
        self.assertIsNone(api_key_from_environment({}))

    def test_missing_key_has_actionable_error(self):
        with self.assertRaises(MaterialsProjectApiKeyError) as error:
            resolve_api_key(None, {})

        self.assertIn("MP_API_KEY", str(error.exception))


class TestMaterialSummary(unittest.TestCase):
    def test_summary_reads_mappings_and_documents(self):
        from_mapping = MaterialSummary.from_document(_document())
        self.assertEqual(from_mapping.material_id, "mp-149")
        self.assertEqual(from_mapping.formula, "Si2")
        self.assertEqual(from_mapping.nsites, 2)
        self.assertAlmostEqual(from_mapping.band_gap, 0.61)
        self.assertTrue(from_mapping.is_stable)

        class Document:
            material_id = "mp-2"
            formula_pretty = "Fe2O3"

        from_document = MaterialSummary.from_document(Document())
        self.assertEqual(from_document.material_id, "mp-2")
        self.assertEqual(from_document.formula, "Fe2O3")
        self.assertIsNone(from_document.volume)
        self.assertIsNone(from_document.is_stable)

    def test_summary_requires_a_material_id(self):
        with self.assertRaises(ValueError):
            MaterialSummary.from_document({})

    def test_summary_serialises_to_json_compatible_values(self):
        self.assertEqual(
            MaterialSummary.from_document(_document()).to_dict()["material_id"],
            "mp-149",
        )


class TestSearchMaterials(unittest.TestCase):
    def test_selectors_are_translated_and_limit_is_applied(self):
        client = FakeClient([_document(material_id=f"mp-{index}") for index in range(1, 6)])

        summaries = search_materials(
            formula="Li*O",
            chemsys="Li-O",
            elements=["Li", "O"],
            material_ids=["mp-1"],
            is_stable=True,
            theoretical=False,
            limit=3,
            client=client,
        )

        self.assertEqual([item.material_id for item in summaries], ["mp-1", "mp-2", "mp-3"])
        self.assertEqual(client.search_kwargs["formula"], "Li*O")
        self.assertEqual(client.search_kwargs["chemsys"], "Li-O")
        self.assertEqual(client.search_kwargs["elements"], ["Li", "O"])
        self.assertEqual(client.search_kwargs["material_ids"], ["mp-1"])
        self.assertTrue(client.search_kwargs["is_stable"])
        self.assertFalse(client.search_kwargs["theoretical"])
        self.assertEqual(list(client.search_kwargs["fields"]), list(DEFAULT_SUMMARY_FIELDS))

    def test_requested_fields_always_include_the_material_id(self):
        client = FakeClient()

        search_materials(formula="Si", fields=["formula_pretty"], client=client)

        self.assertEqual(client.search_kwargs["fields"], ["material_id", "formula_pretty"])

    def test_search_without_a_selector_is_rejected(self):
        with self.assertRaises(ValueError) as error:
            search_materials(client=FakeClient())

        self.assertIn("selector", str(error.exception))

    def test_search_rejects_a_non_positive_limit(self):
        with self.assertRaises(ValueError):
            search_materials(formula="Si", limit=0, client=FakeClient())

    def test_search_without_the_client_never_touches_the_network(self):
        with patch.dict(sys.modules, {"mp_api": None, "mp_api.client": None}):
            with self.assertRaises(MaterialsProjectUnavailableError):
                search_materials(formula="Si")


class TestDownloadMaterial(unittest.TestCase):
    def test_download_returns_the_structure_and_its_summary(self):
        client = FakeClient()

        material = download_material("mp-149", client=client)

        self.assertEqual(material.material_id, "mp-149")
        self.assertEqual(material.summary.formula, "Si2")
        self.assertEqual(len(material.structure), 2)
        self.assertEqual(client.requested_ids, ["mp-149"])
        self.assertEqual(client.search_kwargs["material_ids"], ["mp-149"])

    def test_download_without_a_structure_raises_lookup_error(self):
        with self.assertRaises(LookupError):
            download_material("mp-999", client=FakeClient(structure=None))

    def test_download_tolerates_a_missing_summary_document(self):
        material = download_material("mp-149", client=FakeClient(documents=[]))

        self.assertEqual(material.summary.material_id, "mp-149")
        self.assertIsNone(material.summary.formula)


class TestMaterialFiles(unittest.TestCase):
    def test_material_directory_layout(self):
        self.assertEqual(material_directory(Path("out"), "mp-149"), Path("out/mp-149/STRU"))
        self.assertEqual(
            material_directory(Path("out"), "mp-149", fmt="poscar"),
            Path("out/mp-149/POSCAR"),
        )
        self.assertEqual(
            material_directory(Path("out"), "mp-149", fmt="cif"),
            Path("out/mp-149/structure.cif"),
        )

    def test_material_directory_rejects_an_unknown_format(self):
        with self.assertRaises(ValueError):
            material_directory(Path("out"), "mp-149", fmt="gen")

    def test_write_material_structure_creates_directories_and_stru(self):
        material = download_material("mp-149", client=FakeClient())

        with tempfile.TemporaryDirectory() as temporary:
            destination = material_directory(Path(temporary), "mp-149")
            written = write_material_structure(material, destination)
            text = written.read_text()

        self.assertIn("ATOMIC_SPECIES", text)
        self.assertIn("LATTICE_VECTORS", text)
        self.assertIn("Si", text)

    def test_write_material_structure_writes_other_formats(self):
        material = download_material("mp-149", client=FakeClient())

        with tempfile.TemporaryDirectory() as temporary:
            destination = material_directory(Path(temporary), "mp-149", fmt="cif")
            written = write_material_structure(material, destination, fmt="cif")

            self.assertTrue(written.is_file())

    def test_abacus_structure_keeps_the_material_id_in_metadata(self):
        material = download_material("mp-149", client=FakeClient())

        structure = material.to_abacus_structure()

        self.assertEqual(structure.metadata["material_id"], "mp-149")
        self.assertEqual(structure.metadata["source"], "materials-project")


if __name__ == "__main__":
    unittest.main()
