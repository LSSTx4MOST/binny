"""Tests for the shipped 4MOST CRS BG survey preset."""

from __future__ import annotations

import numpy as np
import pytest

from binny import NZTomography, list_survey_configs

PRESET = "4most_crs_bg"


@pytest.fixture(scope="module")
def tomo():
    return NZTomography().build_survey_bins(
        PRESET,
        role="source",
        sample="bg",
        include_tomo_metadata=True,
        include_survey_metadata=True,
    )


def test_preset_is_listed():
    """Tests that the preset is discovered from the shipped configs."""
    assert PRESET in list_survey_configs()


def test_config_contents():
    """Tests the raw YAML: one source/bg spec-z entry with the Smail fit."""
    cfg = NZTomography.load_survey_config(PRESET)
    assert cfg["name"] == PRESET
    assert len(cfg["tomography"]) == 1
    entry = cfg["tomography"][0]
    assert (entry["role"], entry["sample"], entry["kind"]) == ("source", "bg", "specz")
    assert entry["nz"]["model"] == "smail"
    p = entry["nz"]["params"]
    np.testing.assert_allclose([p["z0"], p["alpha"], p["beta"]], [1.95545983e-02, 9.43, 0.928])


def test_builds_without_selectors():
    """Tests that the single entry is selected when no selectors are given."""
    assert len(NZTomography().build_survey_bins(PRESET).bins) == 5


def test_grid_and_edges(tomo):
    """Tests the 0-0.5 (dz=0.01) grid and the five equidistant edges."""
    np.testing.assert_allclose(tomo.z, np.linspace(0.0, 0.5, 51))
    np.testing.assert_allclose(tomo.tomo_meta["bins"]["bin_edges"], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_parent_nz_normalized_and_peak(tomo):
    """Tests the parent n(z) integrates to 1 and peaks at z0 (alpha/beta)^(1/beta) ~ 0.24."""
    np.testing.assert_allclose(np.trapezoid(tomo.nz, tomo.z), 1.0, rtol=1e-3)
    assert tomo.z[np.argmax(tomo.nz)] == pytest.approx(0.24, abs=0.011)


def test_bins_are_normalized_tophats(tomo):
    """Tests each spec-z bin integrates to 1 and is zero outside its edges."""
    edges = tomo.tomo_meta["bins"]["bin_edges"]
    half_dz = 0.5 * (tomo.z[1] - tomo.z[0])
    for i, key in enumerate(sorted(tomo.bins)):
        curve = tomo.bins[key]
        np.testing.assert_allclose(np.trapezoid(curve, tomo.z), 1.0, rtol=1e-3)
        outside = (tomo.z < edges[i] - half_dz) | (tomo.z > edges[i + 1] + half_dz)
        assert np.all(curve[outside] == 0.0)


def test_footprint_area_is_consistent(tomo):
    """Tests the entry's footprint area matches survey_meta (nominal + overlap scenario)."""
    footprint = tomo.survey_meta["survey_meta"]["footprint"]
    area = tomo.spec["sample_properties"]["footprint"]["survey_area"]
    assert footprint["nominal"]["survey_area"] == area
    assert footprint["overlap_scenarios"]["crs_bg_lsst"]["survey_area"] == area
