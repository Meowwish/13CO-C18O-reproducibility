#!/usr/bin/env python3
"""Transforms and production queries for the v11 adaptive local surrogate."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


RAW_INPUT_NAMES = (
    "nH_cm3", "NH_cm2", "dVdr_km_s_pc", "x12C", "x13C", "x16O", "x18O",
)
OUTPUT_NAMES = (
    "X_CO", "X_13CO", "X_C18O",
    "L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0",
)
TARGET_NAMES = (
    "log10_f_CO",
    "log10_f_13CO",
    "log10_f_C18O",
    "log10_L_CO_1_0",
    "log10_L_13CO_1_0",
    "log10_L_C18O_1_0",
)
FEATURE_NAMES = (
    "log10_nH",
    "log10_NH",
    "log10_dVdr",
    "log10_x12C",
    "log10_x16O",
    "log10_12C_over_13C",
    "log10_16O_over_18O",
    "log10_nH_NH",
    "log10_NH_x12C",
    "log10_NH_x13C",
    "log10_NH_x16O",
    "log10_NH_x18O",
    "log10_NH_over_dVdr",
    "log10_NH_x12C_over_dVdr",
    "log10_NH_x13C_over_dVdr",
    "log10_NH_x18O_over_dVdr",
)


def _raw_log_features(raw_inputs: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_inputs, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(RAW_INPUT_NAMES):
        raise ValueError(
            f"raw_inputs must have shape (N, {len(RAW_INPUT_NAMES)}), got {raw.shape}"
        )
    if np.any(~np.isfinite(raw)):
        raise ValueError("All seven raw inputs must be finite")
    n_h, n_col, dvdr, x12, x13, x16, x18 = raw.T
    if np.any(np.column_stack((n_h, n_col, dvdr, x12, x16)) <= 0.0):
        raise ValueError("nH, NH, dV/dr, x12C, and x16O must be positive")
    if np.any(x13 < 0.0) or np.any(x18 < 0.0):
        raise ValueError("x13C and x18O must be non-negative")
    return np.column_stack((
        np.log10(n_h),
        np.log10(n_col),
        np.log10(dvdr),
        np.log10(x12),
        np.log10(x16),
        np.log10(x12 / np.maximum(x13, 1.0e-30)),
        np.log10(x16 / np.maximum(x18, 1.0e-30)),
    ))


def raw_inputs_to_features(raw_inputs: np.ndarray) -> np.ndarray:
    base = _raw_log_features(raw_inputs)
    log_nh, log_ncol, log_dvdr, log_x12, log_x16, log_c_ratio, log_o_ratio = base.T
    log_x13 = log_x12 - log_c_ratio
    log_x18 = log_x16 - log_o_ratio
    return np.column_stack((
        base,
        log_nh + log_ncol,
        log_ncol + log_x12,
        log_ncol + log_x13,
        log_ncol + log_x16,
        log_ncol + log_x18,
        log_ncol - log_dvdr,
        log_ncol + log_x12 - log_dvdr,
        log_ncol + log_x13 - log_dvdr,
        log_ncol + log_x18 - log_dvdr,
    ))


def elemental_limits(raw_inputs: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_inputs, dtype=float)
    _, _, _, x12, x13, x16, x18 = raw.T
    return np.column_stack((
        np.minimum(x12, x16),
        np.minimum(x13, x16),
        np.minimum(x12, x18),
    ))


def transform_outputs(raw_inputs: np.ndarray, outputs: np.ndarray) -> np.ndarray:
    values = np.asarray(outputs, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(OUTPUT_NAMES):
        raise ValueError(f"outputs must have shape (N, {len(OUTPUT_NAMES)})")
    limits = elemental_limits(raw_inputs)
    fractions = np.clip(values[:, :3] / np.maximum(limits, 1.0e-60), 1.0e-40, 1.0)
    luminosity = np.maximum(values[:, 3:], 1.0e-60)
    return np.column_stack((np.log10(fractions), np.log10(luminosity)))


def inverse_transform_outputs(
    raw_inputs: np.ndarray,
    targets: np.ndarray,
    target_clips: np.ndarray,
) -> np.ndarray:
    target = np.clip(
        np.asarray(targets, dtype=float),
        np.asarray(target_clips, dtype=float)[:, 0],
        np.asarray(target_clips, dtype=float)[:, 1],
    )
    limits = elemental_limits(raw_inputs)
    abundance = np.minimum(limits * 10.0 ** target[:, :3], limits)
    luminosity = 10.0 ** target[:, 3:]
    return np.column_stack((abundance, luminosity))


class AdaptiveLocalActualAbundanceSurrogate:
    """Error-gated v11 surrogate for actual isotope abundances."""

    def __init__(self, path: str | Path, require_validated: bool = True):
        self.path = Path(path).expanduser().resolve()
        self.bundle = joblib.load(self.path)
        self.validation_passed = bool(self.bundle.get("validation_passed", False))
        if require_validated and not self.validation_passed:
            raise RuntimeError(f"v11 surrogate failed validation: {self.path}")

    def query(
        self,
        raw_inputs: np.ndarray,
        chunk_size: int = 1_000_000,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        raw = np.asarray(raw_inputs, dtype=float)
        features = raw_inputs_to_features(raw)
        lower = np.asarray(self.bundle["feature_lower"], dtype=float)
        scale = np.asarray(self.bundle["feature_scale"], dtype=float)
        normalized = np.asarray((features - lower) / scale, dtype=np.float32)
        feature_minimum = np.asarray(self.bundle["feature_minimum"], dtype=float)
        feature_maximum = np.asarray(self.bundle["feature_maximum"], dtype=float)
        within_axis = np.all(
            (features >= feature_minimum) & (features <= feature_maximum), axis=1
        )

        raw_base = _raw_log_features(raw)
        center = np.asarray(self.bundle["coverage_center"], dtype=float)
        inverse_covariance = np.asarray(
            self.bundle["coverage_inverse_covariance"], dtype=float
        )
        delta = raw_base - center
        distance2 = np.einsum("ij,jk,ik->i", delta, inverse_covariance, delta)
        within_joint = distance2 <= float(self.bundle["coverage_distance2_limit"])

        predicted_targets = np.empty((raw.shape[0], len(TARGET_NAMES)), dtype=float)
        for start in range(0, raw.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), raw.shape[0])
            local = normalized[start:stop]
            base = np.column_stack([
                self.bundle["base_models"][name].predict(local)
                for name in TARGET_NAMES
            ])
            strategy = self.bundle.get("selected_candidate", "base")
            if strategy in {"corrected", "ratio_corrected"}:
                augmented = np.column_stack((local, base)).astype(np.float32)
                scales = np.asarray(self.bundle["correction_scales"], dtype=float)
                for column, (name, scale) in enumerate(zip(TARGET_NAMES, scales)):
                    if scale != 0.0:
                        base[:, column] += scale * self.bundle[
                            "correction_models"
                        ][name].predict(augmented)
            if strategy == "ratio_corrected":
                augmented = np.column_stack((local, base)).astype(np.float32)
                ratio_correction = self.bundle["ratio_correction_model"].predict(
                    augmented
                )
                ratio_correction *= float(self.bundle["ratio_correction_scale"])
                base[:, 4] += 0.5 * ratio_correction
                base[:, 5] -= 0.5 * ratio_correction
            predicted_targets[start:stop] = base

        physical = inverse_transform_outputs(
            raw, predicted_targets, self.bundle["target_clips"]
        )
        outputs = {
            name: physical[:, column] for column, name in enumerate(OUTPUT_NAMES)
        }
        reliable = within_axis & within_joint & self.validation_passed
        diagnostics = {
            "within_axis_bounds": within_axis,
            "within_joint_coverage": within_joint,
            "coverage_distance2": distance2,
            "reliable": reliable,
        }
        return outputs, diagnostics


__all__ = [
    "FEATURE_NAMES",
    "AdaptiveLocalActualAbundanceSurrogate",
    "OUTPUT_NAMES",
    "RAW_INPUT_NAMES",
    "TARGET_NAMES",
    "elemental_limits",
    "inverse_transform_outputs",
    "raw_inputs_to_features",
    "transform_outputs",
]
