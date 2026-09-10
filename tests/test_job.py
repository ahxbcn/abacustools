"""Tests for ABACUS job preparation and diagnostics."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import yaml

from abacustools.core.config import CONFIG
from abacustools.core.input_prep import InputPreparationError, InputPreparer
from abacustools.core.job import status_job, validate_job
from abacustools.io.abacus import IsEnabled, ReadInput
from abacustools.main import main


STRU = """\
ATOMIC_SPECIES
H 1.0

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
3 0 0
0 3 0
0 0 3

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
"""


STRU_WITH_ORBITAL = """\
ATOMIC_SPECIES
H 1.0 H.upf

NUMERICAL_ORBITAL
H.orb

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
3 0 0
0 3 0
0 0 3

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0
"""


def _source_and_library(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "water.stru"
    source.write_text(STRU, encoding="utf-8")
    library = tmp_path / "library"
    library.mkdir()
    (library / "H.upf").write_text("pseudo", encoding="utf-8")
    (library / "H.orb").write_text("orbital", encoding="utf-8")
    return source, library


def _variant_library(root: Path) -> Path:
    """Write an SG15-style library with SZ/DZP/TZDP orbital directories."""
    for variant, shell in (("SZ", "1s1p"), ("DZP", "2s2p1d"), ("TZDP", "3s3p2d")):
        directory = root / f"H_{variant}"
        directory.mkdir(parents=True)
        (directory / f"H_gga_7au_100Ry_{shell}.orb").write_text(variant, encoding="utf-8")
    (root / "H.upf").write_text("pseudo", encoding="utf-8")
    return root


def _v2_library(root: Path, rcuts: dict[str, dict[str, float]]) -> Path:
    """Write an orbital-v2.0 library: variant directories plus cutoff indexes."""
    orbitals = root / "Orbitals_v2.0"
    for variant, shell in (("SZ", "1s1p"), ("DZP", "2s2p1d"), ("TZDP", "3s3p2d")):
        directory = orbitals / f"H_{variant}"
        directory.mkdir(parents=True)
        for radius in (7, 8, 10):
            (directory / f"H_gga_{radius}au_100Ry_{shell}.orb").write_text(
                variant, encoding="utf-8"
            )
    for variant, index in rcuts.items():
        index_path = root / f"Orbitals_v2.0_{variant}_E100_StandardRcut.json"
        index_path.write_text(json.dumps(index), encoding="utf-8")
    (root / "H.upf").write_text("pseudo", encoding="utf-8")
    return orbitals


STRU_TWO_ELEMENTS = """\
ATOMIC_SPECIES
H 1.0
O 15.999

LATTICE_CONSTANT
1.0

LATTICE_VECTORS
5 0 0
0 5 0
0 0 5

ATOMIC_POSITIONS
Cartesian

H
0.0
1
0 0 0

O
0.0
1
1 1 1
"""


def _two_element_library(tmp_path: Path) -> tuple[Path, Path]:
    """Write a library whose two orbitals were generated with different cutoffs."""
    source = tmp_path / "water.stru"
    source.write_text(STRU_TWO_ELEMENTS, encoding="utf-8")
    library = tmp_path / "library"
    library.mkdir()
    (library / "H.upf").write_text("pseudo", encoding="utf-8")
    (library / "O.upf").write_text("pseudo", encoding="utf-8")
    (library / "H_gga_8au_100Ry_2s2p1d.orb").write_text("H", encoding="utf-8")
    (library / "O_gga_7au_150Ry_2s2p1d.orb").write_text("O", encoding="utf-8")
    (library / "ecutwfc.json").write_text('{"H": 60, "O": 60}', encoding="utf-8")
    return source, library


def _upf_text(relativistic: str = "scalar", has_so: str = "F") -> str:
    """Return a minimal UPF 2 file with the requested relativistic metadata."""
    return (
        '<UPF version="2.0.1">\n'
        f'  <PP_HEADER element="H" z_valence="1.0" l_max="0" mesh_size="3" '
        f'relativistic="{relativistic}" has_so="{has_so}"/>\n'
        "  <PP_MESH>\n"
        '    <PP_R type="real" size="3">0.0 1.0 2.0</PP_R>\n'
        '    <PP_RAB type="real" size="3">1.0 1.0 1.0</PP_RAB>\n'
        "  </PP_MESH>\n"
        '  <PP_LOCAL size="3">-4.0 -2.0 -1.0</PP_LOCAL>\n'
        "  <PP_NONLOCAL>\n"
        '    <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="2" '
        'cutoff_radius="1.5" size="3">1.0 2.0 3.0</PP_BETA.1>\n'
        '    <PP_DIJ size="1">1.0</PP_DIJ>\n'
        "  </PP_NONLOCAL>\n"
        '  <PP_RHOATOM size="3">0.1 0.2 0.3</PP_RHOATOM>\n'
        "</UPF>\n"
    )


def test_prepare_writes_complete_lcao_job(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    jobs = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=library,
        orb_path=library,
        kpt=[2, 2, 2],
        copy_resources=True,
        folder_syntax="{x[:-5]}",
    ).run()

    job = jobs[0].path
    assert job.name == "water"
    assert (job / "INPUT").is_file()
    assert (job / "STRU").is_file()
    assert (job / "KPT").is_file()
    assert (job / "H.upf").read_text(encoding="utf-8") == "pseudo"
    assert (job / "H.orb").read_text(encoding="utf-8") == "orbital"
    assert ReadInput(job / "INPUT")["basis_type"] == "lcao"
    assert ReadInput(job / "INPUT")["ks_solver"] == "genelpa"
    assert validate_job(job).valid
    assert status_job(job).state == "ready"


def test_prepare_preserves_template_basis_parameters(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    template = tmp_path / "INPUT.template"
    template.write_text(
        "INPUT_PARAMETERS\nbasis_type lcao\nks_solver custom_solver\n",
        encoding="utf-8",
    )
    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        pp_path=library,
        orb_path=library,
        input_template=template,
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path

    assert ReadInput(job / "INPUT")["ks_solver"] == "custom_solver"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (1, True),
        (0, False),
        ("true", True),
        ("T.", False),
        ("yes", True),
        ("no", False),
        ("0", False),
        ("2", True),
        ("", False),
        (None, False),
        ([0.0, 0.0, 0.1], True),
        ([0, 0], False),
    ],
)
def test_is_enabled(value, expected: bool) -> None:
    assert IsEnabled(value) is expected


def test_prepare_pw_job_drops_orbitals(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    source.write_text(STRU_WITH_ORBITAL, encoding="utf-8")

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        orb_path=library,
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H.upf").is_file()
    assert not (job / "H.orb").exists()
    assert "NUMERICAL_ORBITAL" not in (job / "STRU").read_text(encoding="utf-8")


def test_prepare_lcao_job_keeps_orbitals(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    source.write_text(STRU_WITH_ORBITAL, encoding="utf-8")

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=library,
        orb_path=library,
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H.orb").is_file()
    assert "NUMERICAL_ORBITAL" in (job / "STRU").read_text(encoding="utf-8")


def test_prepare_warns_about_the_default_kpt_mesh(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with pytest.warns(UserWarning, match="no KPT file found"):
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
        ).run()[0].path

    assert (job / "KPT").read_text(encoding="utf-8") == "K_POINTS\n0\nGamma\n1 1 1 0 0 0\n"


def test_prepare_uses_kspacing_instead_of_a_kpt_file(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
            set_params={"kspacing": 0.1},
        ).run()[0].path

    assert not (job / "KPT").exists()


def test_prepare_writes_the_cell_relax_keyword(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        job_type="cell-relax",
        kpt=[1, 1, 1],
    ).run()[0].path

    assert ReadInput(job / "INPUT")["calculation"] == "cell-relax"


def test_prepare_rejects_a_broken_element_index(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    (library / "H.upf").unlink()
    (library / "element.json").write_text('{"H": "H.upf"}', encoding="utf-8")

    with pytest.raises(InputPreparationError, match="element.json"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
            kpt=[1, 1, 1],
        ).run()

    # The output directory itself is created first, but no job is written.
    assert not (tmp_path / "jobs" / "000000").exists()


def test_prepare_rejects_conflicting_basis_options(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with pytest.raises(ValueError, match="conflicts with --set basis_type"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=library,
            orb_path=library,
            set_params={"basis_type": "pw"},
        )


def test_prepare_rejects_unknown_basis_type(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    with pytest.raises(ValueError, match="unsupported basis_type: bogus"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            pp_path=library,
            set_params={"basis_type": "bogus"},
        )


def test_prepare_basis_type_choice_uses_matching_solver(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        pp_path=library,
        kpt=[1, 1, 1],
        set_params={"basis_type": "pw"},
    ).run()[0].path

    inputs = ReadInput(job / "INPUT")
    assert inputs["basis_type"] == "pw"
    assert inputs["ks_solver"] == "dav_subspace"
    assert not (job / "H.orb").exists()


def test_prepare_rejects_a_template_with_another_basis(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    template = tmp_path / "INPUT.template"
    template.write_text("INPUT_PARAMETERS\nbasis_type pw\n", encoding="utf-8")

    with pytest.raises(InputPreparationError, match="use a single basis"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=library,
            orb_path=library,
            input_template=template,
        ).run()


def test_prepare_defaults_to_the_dzp_orbital_variant(tmp_path: Path) -> None:
    source, _ = _source_and_library(tmp_path)
    library = _variant_library(tmp_path / "variance")

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=library,
        orb_path=library,
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H_gga_7au_100Ry_2s2p1d.orb").read_text(encoding="utf-8") == "DZP"
    assert not (job / "H_gga_7au_100Ry_1s1p.orb").exists()
    assert "H_gga_7au_100Ry_2s2p1d.orb" in (job / "STRU").read_text(encoding="utf-8")


def test_prepare_honours_the_requested_orbital_variant(tmp_path: Path) -> None:
    source, _ = _source_and_library(tmp_path)
    library = _variant_library(tmp_path / "variance")

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=library,
        orb_path=library,
        orb_variant="TZDP",
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H_gga_7au_100Ry_3s3p2d.orb").read_text(encoding="utf-8") == "TZDP"


def test_prepare_reports_unavailable_orbital_variants(tmp_path: Path) -> None:
    source, _ = _source_and_library(tmp_path)
    library = _variant_library(tmp_path / "variance")

    with pytest.raises(InputPreparationError, match="available variants: DZP, SZ, TZDP"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=library,
            orb_path=library,
            orb_variant="QZP",
            kpt=[1, 1, 1],
        ).run()


def test_prepare_library_variant_overrides_the_global_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_and_library(tmp_path)
    library = _variant_library(tmp_path / "variance")
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": "test",
            "orb_variant": "SZ",
            "libraries": {
                "test": {"pp": str(library), "orb": str(library), "orb_variant": "TZDP"}
            },
        },
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        library="test",
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H_gga_7au_100Ry_3s3p2d.orb").is_file()


def test_prepare_uses_the_standard_cutoff_index(tmp_path: Path) -> None:
    source, _ = _source_and_library(tmp_path)
    orbitals = _v2_library(
        tmp_path / "sg15",
        {"DZP": {"H": 8, "Others": 10}, "TZDP": {"Others": 7}},
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=tmp_path / "sg15",
        orb_path=orbitals,
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H_gga_8au_100Ry_2s2p1d.orb").is_file()
    assert not (job / "H_gga_10au_100Ry_2s2p1d.orb").exists()

    tzdp = InputPreparer(
        source,
        output_dir=tmp_path / "tzdp-jobs",
        filetype="stru",
        basis="lcao",
        pp_path=tmp_path / "sg15",
        orb_path=orbitals,
        orb_variant="TZDP",
        kpt=[1, 1, 1],
    ).run()[0].path
    # H is not listed for TZDP, so the Others entry selects the 7au orbital.
    assert (tzdp / "H_gga_7au_100Ry_3s3p2d.orb").is_file()


def test_prepare_recovers_from_a_renamed_library_directory(tmp_path: Path) -> None:
    source, _ = _source_and_library(tmp_path)
    _v2_library(tmp_path / "sg15", {"DZP": {"H": 8}})

    with pytest.warns(UserWarning, match="does not exist; using"):
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=tmp_path / "sg15",
            orb_path=tmp_path / "sg15" / "Orbitals",
            kpt=[1, 1, 1],
        ).run()[0].path

    assert (job / "H_gga_8au_100Ry_2s2p1d.orb").is_file()


def test_prepare_reports_ambiguous_library_directories(tmp_path: Path) -> None:
    source, _ = _source_and_library(tmp_path)
    root = tmp_path / "sg15"
    for name in ("Orbitals_a", "Orbitals_b"):
        directory = root / name / "H_DZP"
        directory.mkdir(parents=True)
        (directory / "H_gga_8au_100Ry_2s2p1d.orb").write_text("DZP", encoding="utf-8")
    (root / "H.upf").write_text("pseudo", encoding="utf-8")

    with pytest.raises(InputPreparationError, match="candidate directories: .*Orbitals_a"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=root,
            orb_path=root / "Orbitals",
            kpt=[1, 1, 1],
        ).run()


def test_prepare_ignores_an_unused_element_without_the_variant(tmp_path: Path) -> None:
    """Dojo-NC-SR only ships TZDP orbitals for La, which must not break other jobs."""
    source, _ = _source_and_library(tmp_path)
    orbitals = _v2_library(tmp_path / "dojo", {"DZP": {"H": 8}})
    lanthanum = orbitals / "La_TZDP"
    lanthanum.mkdir()
    (lanthanum / "La_gga_8au_100Ry_3s3p2d.orb").write_text("TZDP", encoding="utf-8")

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=tmp_path / "dojo",
        orb_path=orbitals,
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H_gga_8au_100Ry_2s2p1d.orb").is_file()
    assert not (job / "La_gga_8au_100Ry_3s3p2d.orb").exists()


def test_prepare_lcao_uses_the_largest_orbital_cutoff(tmp_path: Path) -> None:
    source, library = _two_element_library(tmp_path)

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        pp_path=library,
        orb_path=library,
        kpt=[1, 1, 1],
    ).run()[0].path

    # The O orbital was generated with 150 Ry, the H orbital with 100 Ry.
    assert ReadInput(job / "INPUT")["ecutwfc"] == pytest.approx(150.0)


def test_prepare_keeps_a_sufficient_explicit_cutoff(tmp_path: Path) -> None:
    source, library = _two_element_library(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=library,
            orb_path=library,
            kpt=[1, 1, 1],
            set_params={"ecutwfc": 200},
        ).run()[0].path

    assert ReadInput(job / "INPUT")["ecutwfc"] == pytest.approx(200.0)


def test_prepare_warns_about_a_too_small_cutoff(tmp_path: Path) -> None:
    source, library = _two_element_library(tmp_path)

    with pytest.warns(UserWarning, match="below the 150 Ry cutoff"):
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            pp_path=library,
            orb_path=library,
            kpt=[1, 1, 1],
            set_params={"ecutwfc": 100},
        ).run()[0].path

    assert ReadInput(job / "INPUT")["ecutwfc"] == pytest.approx(100.0)


def test_prepare_pw_ignores_the_orbital_cutoff(tmp_path: Path) -> None:
    source, library = _two_element_library(tmp_path)

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        kpt=[1, 1, 1],
    ).run()[0].path

    # A plane-wave job takes the cutoff recommended by the pseudopotentials.
    assert ReadInput(job / "INPUT")["ecutwfc"] == pytest.approx(60.0)


def test_prepare_supports_a_custom_library(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The custom library follows the APNS layout: flat, element-named files."""
    source, library = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": "apns",
            "orb_variant": "DZP",
            "libraries": {"custom": {"pp": str(library), "orb": str(library)}},
        },
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        library="custom",
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H.upf").read_text(encoding="utf-8") == "pseudo"
    assert (job / "H.orb").read_text(encoding="utf-8") == "orbital"


def test_prepare_reports_an_empty_library_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_and_library(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "custom", "libraries": {"custom": {"pp": str(empty), "orb": str(empty)}}},
    )

    with pytest.raises(InputPreparationError, match=f"no matching file in {empty}"):
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            library="custom",
            kpt=[1, 1, 1],
        ).run()


def test_default_config_offers_a_custom_library() -> None:
    from abacustools.core.config import _DEFAULT_CONFIG_FILE

    packaged = yaml.safe_load(Path(_DEFAULT_CONFIG_FILE).read_text(encoding="utf-8"))

    assert "custom" in packaged["resources"]["libraries"]
    assert set(packaged["resources"]["libraries"]["custom"]) == {"pp", "orb"}


def _mapped_variant_library(tmp_path: Path) -> dict:
    """Build an APNS-like library with one directory per orbital set."""
    efficiency = tmp_path / "efficiency"
    efficiency.mkdir()
    (efficiency / "H_gga_7au_100Ry_2s2p1d.orb").write_text("efficiency", encoding="utf-8")
    precision = tmp_path / "precision"
    precision.mkdir()
    (precision / "H_gga_10au_100Ry_3s3p2d.orb").write_text("precision", encoding="utf-8")
    pseudopotentials = tmp_path / "pp"
    pseudopotentials.mkdir()
    (pseudopotentials / "H.upf").write_text("pseudo", encoding="utf-8")
    return {
        "pp": str(pseudopotentials),
        "orb": str(efficiency),
        "orb_variants": {
            "efficiency": str(efficiency),
            "precision": str(precision),
        },
    }


def test_prepare_selects_a_mapped_orbital_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": "apns",
            "orb_variant": "DZP",
            "libraries": {"apns": _mapped_variant_library(tmp_path)},
        },
    )

    # A variant name that is not mapped, such as the SG15-style DZP, keeps the
    # configured orbital directory.
    default_job = InputPreparer(
        source,
        output_dir=tmp_path / "default",
        filetype="stru",
        basis="lcao",
        library="apns",
        kpt=[1, 1, 1],
    ).run()[0].path
    assert (default_job / "H_gga_7au_100Ry_2s2p1d.orb").is_file()

    precision_job = InputPreparer(
        source,
        output_dir=tmp_path / "precision",
        filetype="stru",
        basis="lcao",
        library="apns",
        orb_variant="precision",
        kpt=[1, 1, 1],
    ).run()[0].path
    assert (precision_job / "H_gga_10au_100Ry_3s3p2d.orb").is_file()
    assert not (precision_job / "H_gga_7au_100Ry_2s2p1d.orb").exists()


def test_prepare_honours_a_library_variant_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_and_library(tmp_path)
    library = _mapped_variant_library(tmp_path)
    library["orb_variant"] = "precision"
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "apns", "orb_variant": "DZP", "libraries": {"apns": library}},
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        library="apns",
        kpt=[1, 1, 1],
    ).run()[0].path

    assert (job / "H_gga_10au_100Ry_3s3p2d.orb").is_file()


def test_prepare_warns_about_an_unavailable_explicit_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": "apns",
            "orb_variant": "DZP",
            "libraries": {"apns": _mapped_variant_library(tmp_path)},
        },
    )

    with pytest.warns(UserWarning, match="has no SZ variant"):
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            library="apns",
            orb_variant="SZ",
            kpt=[1, 1, 1],
        ).run()[0].path

    assert (job / "H_gga_7au_100Ry_2s2p1d.orb").is_file()


def test_prepare_keeps_quiet_for_the_configured_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, _ = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": "apns",
            "orb_variant": "DZP",
            "libraries": {"apns": _mapped_variant_library(tmp_path)},
        },
    )

    # The SG15-style default is inherited from the configuration, so falling
    # back to the APNS orbital set must stay silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="lcao",
            library="apns",
            kpt=[1, 1, 1],
        ).run()[0].path

    assert (job / "H_gga_7au_100Ry_2s2p1d.orb").is_file()


def _pseudopotential_library(
    tmp_path: Path, relativistic: str = "scalar", has_so: str = "F"
) -> tuple[Path, Path]:
    """Write a one-element job whose pseudopotential has the given metadata."""
    source, library = _source_and_library(tmp_path)
    (library / "H.upf").write_text(
        _upf_text(relativistic, has_so), encoding="utf-8"
    )
    return source, library


def test_prepare_warns_for_scalar_pseudopotentials_with_nspin4(tmp_path: Path) -> None:
    source, library = _pseudopotential_library(tmp_path)

    with pytest.warns(UserWarning, match="nspin=4 requires pseudopotentials"):
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
            nspin=4,
            kpt=[1, 1, 1],
        ).run()[0].path

    # The warning must not stop the job, which is still a noncollinear one.
    inputs = ReadInput(job / "INPUT")
    assert inputs["nspin"] == 4
    assert inputs["noncolin"] == 1


def test_prepare_accepts_fully_relativistic_pseudopotentials(tmp_path: Path) -> None:
    source, library = _pseudopotential_library(tmp_path, relativistic="full", has_so="T")

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        job = InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
            nspin=4,
            kpt=[1, 1, 1],
        ).run()[0].path

    assert ReadInput(job / "INPUT")["noncolin"] == 1


def test_prepare_checks_spin_orbit_support_only_for_nspin4(tmp_path: Path) -> None:
    source, library = _pseudopotential_library(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        InputPreparer(
            source,
            output_dir=tmp_path / "jobs",
            filetype="stru",
            basis="pw",
            pp_path=library,
            nspin=2,
            kpt=[1, 1, 1],
        ).run()


def test_validate_reports_unknown_keyword_and_missing_resource(tmp_path: Path) -> None:
    job = tmp_path / "job"
    job.mkdir()
    (job / "INPUT").write_text(
        "INPUT_PARAMETERS\ncalculation scf\ngamma_only 1\nnot_a_keyword 1\n",
        encoding="utf-8",
    )
    (job / "STRU").write_text(STRU, encoding="utf-8")

    report = validate_job(job)
    assert not report.valid
    assert {issue.code for issue in report.issues} == {
        "unknown-input",
        "missing-pseudopotential",
    }
    assert validate_job(job, strict=True).valid is False


def test_status_reads_scf_progress_and_convergence(tmp_path: Path) -> None:
    source, library = _source_and_library(tmp_path)
    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="pw",
        pp_path=library,
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path
    output = job / "OUT.ABACUS"
    output.mkdir()
    (output / "running_scf.log").write_text(
        "E_KohnSham = -1.000000\n"
        "density error = 1.0E-4\n"
        "E_KohnSham = -1.100000\n"
        "density error = 1.0E-8\n"
        "charge density convergence is achieved\n",
        encoding="utf-8",
    )

    status = status_job(job)
    assert status.state == "converged"
    assert status.progress["scf_steps"] == 2
    assert status.progress["denergy"] == pytest.approx(-0.1)


def test_prepare_uses_configured_default_library(tmp_path: Path, monkeypatch) -> None:
    source, library = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": library, "orb": library}}},
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path

    assert (job / "H.upf").is_file()
    assert (job / "H.orb").is_file()


def test_prepare_selects_requested_configured_library(tmp_path: Path, monkeypatch) -> None:
    source, library = _source_and_library(tmp_path)
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "H.upf").write_text("selected pseudo", encoding="utf-8")
    (selected / "H.orb").write_text("selected orbital", encoding="utf-8")
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {
            "default": "test",
            "libraries": {
                "test": {"pp": library, "orb": library},
                "selected": {"pp": selected, "orb": selected},
            },
        },
    )

    job = InputPreparer(
        source,
        output_dir=tmp_path / "jobs",
        filetype="stru",
        basis="lcao",
        library="selected",
        kpt=[1, 1, 1],
        copy_resources=True,
    ).run()[0].path

    assert (job / "H.upf").read_text(encoding="utf-8") == "selected pseudo"
    assert (job / "H.orb").read_text(encoding="utf-8") == "selected orbital"


def test_prepare_rejects_unknown_library(tmp_path: Path, monkeypatch) -> None:
    source, _ = _source_and_library(tmp_path)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": None, "orb": None}}},
    )

    with pytest.raises(ValueError, match="unsupported resource library"):
        InputPreparer(source, library="missing")


def test_prepare_reports_unconfigured_library_path(tmp_path: Path, monkeypatch) -> None:
    source, _ = _source_and_library(tmp_path)
    monkeypatch.delenv("ABACUS_PP_PATH", raising=False)
    monkeypatch.delenv("ABACUS_ORB_PATH", raising=False)
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": None, "orb": None}}},
    )

    with pytest.raises(InputPreparationError, match=r"resources\.libraries\.test\.pp"):
        InputPreparer(source, filetype="stru", basis="pw", kpt=[1, 1, 1]).run()


def test_explicit_library_does_not_use_legacy_environment_paths(
    tmp_path: Path, monkeypatch
) -> None:
    source, library = _source_and_library(tmp_path)
    monkeypatch.setenv("ABACUS_PP_PATH", str(library))
    monkeypatch.setenv("ABACUS_ORB_PATH", str(library))
    monkeypatch.setitem(
        CONFIG,
        "resources",
        {"default": "test", "libraries": {"test": {"pp": None, "orb": None}}},
    )

    with pytest.raises(InputPreparationError, match=r"resources\.libraries\.test\.pp"):
        InputPreparer(
            source,
            filetype="stru",
            basis="pw",
            library="test",
            kpt=[1, 1, 1],
        ).run()


def test_job_prepare_help_does_not_expose_resource_paths(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["job", "prepare", "--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--library" in output
    assert "--pp" not in output
    assert "--orb" not in output
