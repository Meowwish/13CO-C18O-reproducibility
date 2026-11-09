#!/usr/bin/env python3
"""Load and validate the full-direct workflow configuration."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config" / "snapshot600_full_direct.json"
CONFIG_ENVIRONMENT_VARIABLE = "CHEMTABLE_V4_CONFIG"


def _positive(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("Only full-direct configuration schema_version=1 is supported")

    required_sections = (
        "paths", "snapshot", "isotope_fields", "shielding_column", "cloud",
        "chemistry", "execution", "projection", "mock_observation", "diagnostics",
    )
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"Configuration is missing sections: {', '.join(missing)}")
    if config["snapshot"].get("stellar_yield_mode") not in {
        "all", "slug_only", "old_only",
    }:
        raise ValueError("snapshot.stellar_yield_mode must be all, slug_only, or old_only")

    for isotope in ("12C", "13C", "16O", "18O"):
        entries = config["isotope_fields"].get(isotope)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"isotope_fields.{isotope} must be a non-empty list")
        for entry in entries:
            if "field" not in entry:
                raise ValueError(f"isotope_fields.{isotope} entry lacks field: {entry}")
            if "factor" not in entry and not {"numerator", "denominator"} <= entry.keys():
                raise ValueError(
                    f"isotope_fields.{isotope} entry needs factor or numerator/denominator"
                )
            if "denominator" in entry:
                _positive(entry["denominator"], f"isotope_fields.{isotope}.denominator")

    chemistry = config["chemistry"]
    if chemistry.get("network") != "GOW":
        raise ValueError("The full-direct workflow currently supports only network=GOW")
    solver = chemistry["setChemEq"]
    _positive(solver["relative_tolerance"], "chemistry.setChemEq.relative_tolerance")
    _positive(solver["small_abundance"], "chemistry.setChemEq.small_abundance")
    _positive(solver["maximum_time_primary_s"], "chemistry.setChemEq.maximum_time_primary_s")
    _positive(solver["maximum_time_retry_s"], "chemistry.setChemEq.maximum_time_retry_s")
    if solver["tEqGuess_s"] is not None:
        _positive(solver["tEqGuess_s"], "chemistry.setChemEq.tEqGuess_s")
    if config["cloud"]["escape_probability_geometry"] not in {"LVG", "sphere", "slab"}:
        raise ValueError("Unsupported escape_probability_geometry")
    shielding = config["shielding_column"]
    shielding_mode = shielding["mode"]
    if shielding_mode not in {"projected_face_on", "jeans", "local_sobolev"}:
        raise ValueError(
            "shielding_column.mode must be projected_face_on, jeans, or local_sobolev"
        )
    if shielding_mode in {"jeans", "local_sobolev"}:
        _positive(shielding.get("jeans_gamma"), "shielding_column.jeans_gamma")
    if shielding_mode == "local_sobolev":
        _positive(
            shielding.get("minimum_smoothing_length_fraction"),
            "shielding_column.minimum_smoothing_length_fraction",
        )
    if config["isotope_fields"]["oxygen_renormalization"]["weighting"] != "gas_mass":
        raise ValueError("Only gas-mass-weighted oxygen renormalization is implemented")
    if config["chemistry"].get("include_C18O_fractionation", False):
        raise ValueError("C18O fractionation is recorded but is not implemented")
    if config["projection"]["unconverged_particle_policy"] != "zero":
        raise ValueError("Only the zero unconverged-particle projection policy is implemented")
    mock = config["mock_observation"]
    _positive(mock["beam_pc"], "mock_observation.beam_pc")
    _positive(mock["hex_size_pc"], "mock_observation.hex_size_pc")
    _positive(mock["detection_snr_threshold"], "mock_observation.detection_snr_threshold")
    if not mock["noise_levels_K_km_s"]:
        raise ValueError("mock_observation.noise_levels_K_km_s cannot be empty")

    for mode in ("primary", "retry"):
        run = config["execution"][mode]
        _positive(run["chunk_size"], f"execution.{mode}.chunk_size")
        _positive(run["workers"], f"execution.{mode}.workers")
        if not run["damp_factors"]:
            raise ValueError(f"execution.{mode}.damp_factors cannot be empty")
        for damp in run["damp_factors"]:
            _positive(damp, f"execution.{mode}.damp_factors")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_config(path: Path, stack: tuple[Path, ...] = ()) -> tuple[dict[str, Any], bool]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Configuration inheritance cycle: {chain}")
    raw = json.loads(path.read_text(encoding="ascii"))
    parent_value = raw.pop("extends", None)
    if parent_value is None:
        return raw, False
    parent = Path(parent_value).expanduser()
    if not parent.is_absolute():
        parent = (path.parent / parent).resolve()
    base, _ = _read_config(parent, (*stack, path))
    return _deep_merge(base, raw), True


def load_config(path: str | os.PathLike[str] | None = None):
    selected = Path(
        path or os.environ.get(CONFIG_ENVIRONMENT_VARIABLE, DEFAULT_CONFIG_PATH)
    ).expanduser().resolve()
    config, inherited = _read_config(selected)
    validate_config(config)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    digest_source = canonical.encode("ascii") if inherited else selected.read_bytes()
    digest = hashlib.sha256(digest_source).hexdigest()
    return config, selected, digest, canonical


def config_path(config: dict[str, Any], key: str) -> Path:
    value = Path(config["paths"][key]).expanduser()
    return value if value.is_absolute() else (ROOT / value).resolve()


def provenance_attributes(config_path_value: Path, digest: str, canonical: str):
    return {
        "workflow_config_path": str(config_path_value),
        "workflow_config_sha256": digest,
        "workflow_config_schema_version": 1,
        "workflow_config_json": canonical,
    }


def write_provenance(attributes, config_path_value: Path, digest: str, canonical: str):
    for name, value in provenance_attributes(config_path_value, digest, canonical).items():
        attributes[name] = value


if __name__ == "__main__":
    configuration, selected_path, sha256, _ = load_config()
    print(json.dumps({
        "path": str(selected_path),
        "sha256": sha256,
        "schema_version": configuration["schema_version"],
        "run_name": configuration["run_name"],
    }, indent=2))
