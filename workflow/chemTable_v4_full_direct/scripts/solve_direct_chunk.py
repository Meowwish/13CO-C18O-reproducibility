#!/usr/bin/env python3
"""Run full DESPOTIC/GOW solutions for one prepared particle chunk."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import traceback
import warnings

from workflow_config import config_path, load_config, write_provenance


ROOT = Path(__file__).resolve().parents[1]
CONFIG, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL = load_config()
RUN_MODE = os.environ.get("DIRECT_RUN_MODE", "primary").strip().lower()
if RUN_MODE not in {"primary", "retry"}:
    raise ValueError(f"DIRECT_RUN_MODE must be primary or retry, got {RUN_MODE!r}")
RUN = CONFIG["execution"][RUN_MODE]
THREAD_LIMITS = CONFIG["execution"]["thread_limits"]
for environment_name, value in THREAD_LIMITS.items():
    os.environ.setdefault(environment_name, str(value))
os.environ.setdefault("DESPOTIC_HOME", str(ROOT))
os.environ["VISSER_CO_SHIELDING_DIR"] = str(
    config_path(CONFIG, "visser_shielding_dir")
)

import astropy.units as u
import h5py
import numpy as np


INPUT_PATH = config_path(
    CONFIG, "direct_input" if RUN_MODE == "primary" else "retry_input"
)
OUTPUT_DIR = config_path(
    CONFIG, "primary_chunk_dir" if RUN_MODE == "primary" else "retry_chunk_dir"
)
CLOUD_FILE = config_path(CONFIG, "cloud_file")
ARRAY_INDEX = int(os.environ.get("PBS_ARRAY_INDEX", os.environ.get("DIRECT_CHUNK_INDEX", "0")))
CHUNK_SIZE = int(RUN["chunk_size"])
ALLOCATED_WORKERS = int(os.environ.get("PBS_NCPUS", RUN["workers"]))
WORKERS = min(int(RUN["workers"]), ALLOCATED_WORKERS)
CHEMISTRY = CONFIG["chemistry"]
SOLVER = CHEMISTRY["setChemEq"]
CLOUD = CONFIG["cloud"]
SMALL_ABUNDANCE = float(SOLVER["small_abundance"])
MAX_TIME_S = float(SOLVER[
    "maximum_time_primary_s" if RUN_MODE == "primary" else "maximum_time_retry_s"
])
DAMP_FACTORS = tuple(float(value) for value in RUN["damp_factors"])
DVDR_CGS = (u.km / u.s / u.pc).to("1/s")
REFERENCE = CHEMISTRY["reference_abundances_per_H_nucleus"]
INITIALIZATION = CHEMISTRY["molecular_initialization"]
warnings.filterwarnings("ignore")

OUTPUT_NAMES = (
    "Tg_K", "Td_K", "X_H2", "X_CO", "X_13CO", "X_C18O",
    "L_CO_1_0", "L_13CO_1_0", "L_C18O_1_0",
    "tau_CO_1_0", "tau_13CO_1_0", "tau_C18O_1_0",
    "Tex_CO_1_0_K", "Tex_13CO_1_0_K", "Tex_C18O_1_0_K",
    "TB_CO_1_0_K_km_s", "TB_13CO_1_0_K_km_s", "TB_C18O_1_0_K_km_s",
)
INPUT_NAMES = (
    "nH_cm3", "NH_cm2", "dVdr_km_s_pc", "x12C", "x13C", "x16O", "x18O",
)


def set_emitter(cloud, name, abundance):
    emitters = {key.lower(): value for key, value in cloud.emitters.items()}
    if name.lower() in emitters:
        emitters[name.lower()].abundance = abundance
    else:
        cloud.addEmitter(name, abundance, energySkip=True)


def make_cloud(row, damp_factor):
    from despotic import cloud
    from despotic.chemistry.GOW import GOW

    n_h, n_col, dvdr, x12, x13, x16, x18 = row
    model = cloud(str(CLOUD_FILE))
    model.comp.xHe = float(CLOUD["helium_abundance_per_H_nucleus"])
    model.rad.TradDust = float(CLOUD["dust_radiation_temperature_K"])
    model.rad.chi = float(CLOUD["chi_Draine"])
    model.rad.ionRate = float(CLOUD["cosmic_ray_ionization_rate_s-1"])
    model.dampFactor = damp_factor
    model.Td = float(CLOUD["initial_Td_K"])
    model.Tg = float(CLOUD["initial_Tg_K"])
    model.nH = n_h
    model.colDen = n_col
    model.dVdr = dvdr * DVDR_CGS
    model.noWarn = True

    f_co = min(x12 / REFERENCE["12C"], x16 / REFERENCE["16O"])
    f_13 = min(x13 / REFERENCE["13C"], x16 / REFERENCE["16O"])
    f_18 = min(x12 / REFERENCE["12C"], x18 / REFERENCE["18O"])
    cap = float(INITIALIZATION["elemental_seed_cap_fraction"])
    combined_cap = float(INITIALIZATION["combined_isotopologue_cap_fraction"])
    minimum = float(INITIALIZATION["minimum_emitter_abundance"])
    co_seed = min(max(0.0, REFERENCE["CO"] * f_co), cap * x12, cap * x16)
    c18_seed = min(
        max(0.0, REFERENCE["C18O"] * f_18),
        cap * x18,
        max(0.0, combined_cap * x12 - co_seed),
    )
    c13_seed = min(
        max(0.0, REFERENCE["13CO"] * f_13),
        cap * x13,
        max(0.0, combined_cap * x16 - co_seed),
    )
    set_emitter(model, "co", max(co_seed, minimum))
    set_emitter(model, "13co", max(c13_seed, minimum))
    set_emitter(model, "c18o", max(c18_seed, minimum))
    for emitter in model.emitters.values():
        try:
            emitter.setLevPop(model, thin=True, noClump=True)
            emitter.data.noWarn = True
        except Exception:
            pass

    c12c13 = x12 / max(x13, 1.0e-30)
    o16o18 = x16 / max(x18, 1.0e-30)
    info = {
        "xC": x12,
        "xO": x16,
        "c12c13": c12c13,
        "o16o18": o16o18,
        "include13CFractionation": bool(CHEMISTRY["include_13C_fractionation"]),
        "visserBCO": float(CHEMISTRY["visser_b_CO_km_s"]),
        "noClump": bool(CLOUD["no_clumping"]),
    }
    model.chemnetwork = GOW(model, info=info)
    return model


def line_diagnostics(model, species):
    records = model.lineLum(
        species,
        transition=[np.array([1]), np.array([0])],
        noClump=bool(CLOUD["no_clumping"]),
        escapeProbGeom=CLOUD["escape_probability_geometry"],
    )
    record = records[0]
    return (
        float(record["lumPerH"]),
        float(record["tau"]),
        float(record["Tex"]),
        float(record["intTB"]),
    )


def solve_one(task):
    local_index, row = task
    from despotic.chemistry.GOW import GOW

    last_error = ""
    last_traceback = ""
    for attempt, damp in enumerate(DAMP_FACTORS, start=1):
        try:
            model = make_cloud(row, damp)
            model.setDustTempEq()
            converged = model.setChemEq(
                tEqGuess=SOLVER["tEqGuess_s"],
                network=GOW,
                evolveTemp=SOLVER["evolve_temperature"],
                tempEqParam={
                    "noClump": bool(CLOUD["no_clumping"]),
                    "escapeProbGeom": CLOUD["escape_probability_geometry"],
                },
                addEmitters=bool(SOLVER["add_emitters"]),
                maxTime=MAX_TIME_S,
                tol=float(SOLVER["relative_tolerance"]),
                smallabd=SMALL_ABUNDANCE,
                convList=SOLVER["convergence_species"],
                maxTempIter=int(SOLVER["maximum_temperature_iterations"]),
            )
            if not converged or not np.isfinite(model.Tg) or model.Tg <= 0.0:
                raise RuntimeError("chemical/thermal solve did not converge")
            abundance = model.chemnetwork.abundances
            molecular = np.array([
                float(abundance["H2"]),
                float(abundance["CO"]),
                float(abundance["13CO"]),
                float(abundance["C18O"]),
            ])
            if np.any(~np.isfinite(molecular)) or np.any(molecular < -1.0e-12):
                raise RuntimeError("non-physical molecular abundance")
            molecular = np.maximum(molecular, 0.0)
            co_line = line_diagnostics(model, "co")
            c13_line = line_diagnostics(model, "13co")
            c18_line = line_diagnostics(model, "c18o")
            values = np.array([
                float(model.Tg), float(model.Td), *molecular,
                co_line[0], c13_line[0], c18_line[0],
                co_line[1], c13_line[1], c18_line[1],
                co_line[2], c13_line[2], c18_line[2],
                co_line[3], c13_line[3], c18_line[3],
            ], dtype=float)
            if np.any(~np.isfinite(values)):
                raise RuntimeError("non-finite DESPOTIC output")
            return local_index, 1, attempt, values, ""
        except Exception as error:
            last_error = f"{type(error).__name__}: {error}"
            last_traceback = traceback.format_exc(limit=8).strip()
    return local_index, -1, len(DAMP_FACTORS), np.full(len(OUTPUT_NAMES), np.nan), (
        last_error + " | " + last_traceback
    )


def input_tasks(arrays):
    count = arrays[INPUT_NAMES[0]].size
    for index in range(count):
        yield index, tuple(float(arrays[name][index]) for name in INPUT_NAMES)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"direct_chunk_{ARRAY_INDEX:04d}.h5"
    if output_path.exists():
        with h5py.File(output_path, "r") as existing:
            if bool(existing.attrs.get("complete", False)):
                print(f"Complete chunk already exists: {output_path}")
                return

    with h5py.File(INPUT_PATH, "r") as source:
        if not bool(source.attrs.get("complete", False)):
            raise RuntimeError(f"Input is incomplete: {INPUT_PATH}")
        input_config = source.attrs.get("workflow_config_sha256")
        if input_config is not None and input_config != CONFIG_SHA256:
            raise RuntimeError(
                "Input was prepared with a different workflow configuration: "
                f"{input_config} != {CONFIG_SHA256}"
            )
        total = int(source.attrs["active_particle_count"])
        start = ARRAY_INDEX * CHUNK_SIZE
        stop = min(start + CHUNK_SIZE, total)
        if start >= total:
            print(f"Chunk {ARRAY_INDEX} is beyond active row count {total}; no work")
            return
        arrays = {
            name: np.asarray(source[name][start:stop], dtype=float)
            for name in INPUT_NAMES
        }
        particle_index = np.asarray(source["particle_index"][start:stop], dtype=np.int64)

    count = stop - start
    print(
        f"chunk={ARRAY_INDEX} rows={start}:{stop} count={count} "
        f"mode={RUN_MODE} workers={WORKERS} smallabd={SMALL_ABUNDANCE:.1e} "
        f"config={CONFIG_SHA256[:12]}"
    )
    values = np.full((count, len(OUTPUT_NAMES)), np.nan, dtype=float)
    status = np.zeros(count, dtype=np.int8)
    attempts = np.zeros(count, dtype=np.int8)
    errors = Counter()
    complete = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for index, state, attempt, result, error in executor.map(
            solve_one, input_tasks(arrays), chunksize=1
        ):
            status[index] = state
            attempts[index] = attempt
            values[index] = result
            if error:
                errors[error[:500]] += 1
            complete += 1
            if complete % 250 == 0 or complete == count:
                print(
                    f"completed={complete}/{count} "
                    f"converged={np.count_nonzero(status == 1)}"
                )

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["complete"] = False
        write_provenance(
            handle.attrs, CONFIG_PATH, CONFIG_SHA256, CONFIG_CANONICAL
        )
        handle.attrs["run_mode"] = RUN_MODE
        handle.attrs["chunk_index"] = ARRAY_INDEX
        handle.attrs["input_start"] = start
        handle.attrs["input_stop"] = stop
        handle.attrs["input_path"] = str(INPUT_PATH)
        handle.attrs["small_abundance"] = SMALL_ABUNDANCE
        handle.attrs["max_time_s"] = MAX_TIME_S
        handle.attrs["worker_count"] = WORKERS
        handle.attrs["converged_count"] = int(np.count_nonzero(status == 1))
        handle.attrs["failed_count"] = int(np.count_nonzero(status != 1))
        handle.attrs["error_summary_json"] = json.dumps(errors.most_common(20))
        handle.create_dataset("particle_index", data=particle_index, compression="lzf")
        handle.create_dataset("status", data=status, compression="lzf")
        handle.create_dataset("attempt", data=attempts, compression="lzf")
        for column, name in enumerate(OUTPUT_NAMES):
            handle.create_dataset(name, data=values[:, column], compression="lzf", shuffle=True)
        handle.attrs["complete"] = True
        handle.flush()
    os.replace(temporary, output_path)
    print(
        f"Saved {output_path}; converged={np.count_nonzero(status == 1)}/{count}, "
        f"failed={np.count_nonzero(status != 1)}"
    )


if __name__ == "__main__":
    main()
