"""Tests for UPF 1.x and UPF 2 parsing."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from abacustools.io.pseudo import UPF


UPF_FIXTURE = """\
<UPF version="2.0.1">
  <PP_HEADER element="Si" z_valence="4.0" l_max="1" mesh_size="3"/>
  <PP_MESH>
    <PP_R type="real" size="3">0.0 1.0 2.0</PP_R>
    <PP_RAB type="real" size="3">1.0 1.0 1.0</PP_RAB>
  </PP_MESH>
  <PP_LOCAL size="3">-4.0 -2.0 -1.0</PP_LOCAL>
  <PP_NONLOCAL>
    <PP_BETA.1 index="1" angular_momentum="0" cutoff_radius_index="2" cutoff_radius="1.5" size="3">1.0 2.0 3.0</PP_BETA.1>
    <PP_BETA.2 index="2" angular_momentum="1" cutoff_radius_index="2" cutoff_radius="2.0" size="3">4.0 5.0 6.0</PP_BETA.2>
    <PP_DIJ size="4">1.0 0.0 0.0 2.0</PP_DIJ>
  </PP_NONLOCAL>
  <PP_PSWFC>
    <PP_CHI.1 index="1" l="0" label="3S" occupation="2.0" pseudo_energy="-0.4" size="3">0.1 0.2 0.3</PP_CHI.1>
    <PP_CHI.2 index="2" l="1" label="3P" occupation="2.0" pseudo_energy="-0.2" size="3">0.4 0.5 0.6</PP_CHI.2>
  </PP_PSWFC>
  <PP_NLCC size="3">0.3 0.2 0.1</PP_NLCC>
  <PP_RHOATOM size="3">0.1 0.2 0.3</PP_RHOATOM>
</UPF>
"""


LEGACY_UPF_FIXTURE = """\
<PP_INFO>
3S  3  0  2.00  0.000000  1.20  -0.500000
3P  3  1  1.00  0.000000  1.40  -0.200000
</PP_INFO>
<PP_HEADER>
  1.0 Version Number
  Si Element
  4.00000000000 Z valence
  1 Max angular momentum component
  3 Number of points in mesh
  2 2 Number of Wavefunctions, Number of Projectors
</PP_HEADER>
<PP_MESH>
  <PP_R>
    0.0 0.1 0.2
  </PP_R>
  <PP_RAB>
    0.1 0.1 0.1
  </PP_RAB>
</PP_MESH>
<PP_LOCAL>
  -4.0 -2.0 -1.0
</PP_LOCAL>
<PP_NLCC>
  0.3 0.2 0.1
</PP_NLCC>
<PP_NONLOCAL>
  <PP_BETA>
    1 0 Beta L
    2
    1.0 2.0
  </PP_BETA>
  <PP_BETA>
    2 1 Beta L
    2
    3.0 4.0
  </PP_BETA>
  <PP_DIJ>
    3 Number of nonzero Dij
    1 1 10.0
    1 2 20.0
    2 2 30.0
  </PP_DIJ>
</PP_NONLOCAL>
<PP_PSWFC>
3S  0  2.00  Wavefunction
  0.1 0.2 0.3
3P  1  1.00  Wavefunction
  0.4 0.5 0.6
</PP_PSWFC>
<PP_RHOATOM>
  0.1 0.2 0.3
</PP_RHOATOM>
"""


class TestUPF(unittest.TestCase):
    def _write_upf(self, contents: str) -> Path:
        temporary_directory = TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        path = Path(temporary_directory.name) / "test.upf"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_read_upf_radial_data_and_metadata(self) -> None:
        pp = UPF.read_from_file(self._write_upf(UPF_FIXTURE))

        self.assertEqual(pp.element, "Si")
        self.assertEqual(pp.version, "2.0.1")
        self.assertEqual(pp.valence, 4.0)
        self.assertEqual(pp.mesh, 3)
        self.assertEqual(pp.cutoff_radius, 2.0)
        np.testing.assert_allclose(pp.local_potential, [-4.0, -2.0, -1.0])
        np.testing.assert_allclose(pp.dij_matrix, [[1.0, 0.0], [0.0, 2.0]])
        self.assertEqual(pp.pseudo_wavefunctions[1]["label"], "3P")
        np.testing.assert_allclose(pp.pseudo_wavefunctions[1]["data"], [0.4, 0.5, 0.6])
        self.assertEqual(
            [section["tag"] for section in pp.sections],
            [
                "PP_HEADER",
                "PP_MESH",
                "PP_LOCAL",
                "PP_NONLOCAL",
                "PP_PSWFC",
                "PP_NLCC",
                "PP_RHOATOM",
            ],
        )
        mesh_section = pp.sections[1]
        np.testing.assert_allclose(mesh_section["children"][0]["data"], [0.0, 1.0, 2.0])
        self.assertIsNotNone(pp.nlcc)

    def test_rejects_mismatched_declared_data_size(self) -> None:
        contents = UPF_FIXTURE.replace(
            '<PP_RHOATOM size="3">0.1 0.2 0.3</PP_RHOATOM>',
            '<PP_RHOATOM size="3">0.1 0.2</PP_RHOATOM>',
        )

        with self.assertRaisesRegex(ValueError, "PP_RHOATOM.*declares size 3"):
            UPF.read_from_file(self._write_upf(contents))

    def test_uses_numbered_tag_for_nonstandard_projector_index(self) -> None:
        contents = UPF_FIXTURE.replace('index="2" angular_momentum="1"', 'index="*" angular_momentum="1"')
        pp = UPF.read_from_file(self._write_upf(contents))

        self.assertEqual(pp.projectors[1]["index"], 2)

    def test_reads_sectioned_upf_1(self) -> None:
        pp = UPF.read_from_file(self._write_upf(LEGACY_UPF_FIXTURE))

        self.assertEqual(pp.version, "1.0")
        self.assertEqual(pp.element, "Si")
        self.assertEqual(pp.valence, 4.0)
        self.assertEqual(pp.mesh, 3)
        self.assertEqual(pp.cutoff_radius, 1.4)
        self.assertEqual(len(pp.projectors), 2)
        self.assertEqual([projector["data"].size for projector in pp.projectors], [2, 2])
        np.testing.assert_allclose(pp.dij_matrix, [[10.0, 20.0], [20.0, 30.0]])
        self.assertEqual([wfc["label"] for wfc in pp.pseudo_wavefunctions], ["3S", "3P"])
        self.assertEqual(pp.pseudo_wavefunctions[0]["pseudo_energy"], -0.5)
        np.testing.assert_allclose(pp.rhoatom, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(pp.nlcc, [0.3, 0.2, 0.1])


if __name__ == "__main__":
    unittest.main()
