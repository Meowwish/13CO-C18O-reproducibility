"""CO isotopologue shielding from Visser, van Dishoeck & Black (2009).

The distributed tables contain line-shielding factors for several CO
isotopologues as functions of N(12CO) and N(H2).  This module interpolates
logarithmic shielding factors across column density, CO Doppler width,
excitation temperature, and 12C/13C ratio.  Dust attenuation remains the
responsibility of the photoreaction ``av_fac`` in DESPOTIC.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


_B_VALUES = np.array([0.3, 3.0])
_TEX_VALUES = np.array([5.0, 20.0, 50.0, 100.0])
_C12_C13_VALUES = np.array([35.0, 69.0])
_O16_O18_TABLE_RATIO = 557.0
_SPECIES = ("12C16O", "13C16O", "12C18O")


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "visser_co_shielding"


def _read_table(path: Path):
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines()]
    nco_start = lines.index("N(12CO)") + 1
    nh2_header = lines.index("N(H2)")
    species_headers = {name: lines.index(name) for name in (
        "12C16O", "12C17O", "12C18O", "13C16O", "13C17O", "13C18O"
    )}
    first_species = min(species_headers.values())

    nco = np.array([float(value) for value in lines[nco_start:nh2_header]])
    nh2 = np.array([float(value) for value in lines[nh2_header + 1:first_species]])
    arrays = {}
    ordered_headers = sorted((index, name) for name, index in species_headers.items())
    for position, (start, name) in enumerate(ordered_headers):
        stop = ordered_headers[position + 1][0] if position + 1 < len(ordered_headers) else len(lines)
        values = np.fromstring(" ".join(lines[start + 1:stop]), sep=" ")
        expected = nh2.size * nco.size
        if values.size != expected:
            raise ValueError(f"{path}: {name} has {values.size} values, expected {expected}")
        arrays[name] = values.reshape(nh2.size, nco.size)
    return nco, nh2, arrays


class VisserCOShielding:
    """Lazy interpolator over the public Visser et al. shielding tables."""

    def __init__(self, data_dir: str | os.PathLike[str] | None = None):
        self.data_dir = Path(data_dir) if data_dir is not None else Path(
            os.environ.get("VISSER_CO_SHIELDING_DIR", _default_data_dir())
        )
        self._interpolators = self._load_interpolators()

    @staticmethod
    def _filename(b_co: float, tex: float, c12_c13: float) -> str:
        b_code = "03" if np.isclose(b_co, 0.3) else "30"
        return f"shield.{b_code}.{int(tex)}.{int(c12_c13)}-557-36.dat"

    def _load_interpolators(self):
        data = None
        nco_ref = None
        nh2_ref = None
        for ib, b_co in enumerate(_B_VALUES):
            for it, tex in enumerate(_TEX_VALUES):
                for ic, c12_c13 in enumerate(_C12_C13_VALUES):
                    path = self.data_dir / self._filename(b_co, tex, c12_c13)
                    nco, nh2, arrays = _read_table(path)
                    if data is None:
                        nco_ref = nco
                        nh2_ref = nh2
                        data = {
                            species: np.empty((2, 4, 2, nh2.size, nco.size))
                            for species in _SPECIES
                        }
                    elif not (np.array_equal(nco, nco_ref) and np.array_equal(nh2, nh2_ref)):
                        raise ValueError(f"Visser grid mismatch in {path}")
                    for species in _SPECIES:
                        data[species][ib, it, ic] = arrays[species]

        self.nco_bounds = (float(nco_ref[0]), float(nco_ref[-1]))
        self.nh2_bounds = (float(nh2_ref[0]), float(nh2_ref[-1]))
        axes = (
            np.log10(_B_VALUES),
            np.log10(_TEX_VALUES),
            np.log10(_C12_C13_VALUES),
            np.log10(nh2_ref),
            np.log10(nco_ref),
        )
        return {
            species: RegularGridInterpolator(
                axes,
                np.log10(np.clip(values, 1.0e-300, 1.0)),
                bounds_error=False,
                fill_value=None,
            )
            for species, values in data.items()
        }

    def factor(self, species, n12co, nh2, b_co=0.3, tex=20.0, c12_c13=60.0):
        if species not in self._interpolators:
            raise KeyError(f"unsupported CO isotopologue {species!r}")

        arrays = np.broadcast_arrays(n12co, nh2, b_co, tex, c12_c13)
        n12co_arr, nh2_arr, b_arr, tex_arr, ratio_arr = [
            np.asarray(value, dtype=float) for value in arrays
        ]
        points = np.column_stack((
            np.log10(np.clip(b_arr.ravel(), _B_VALUES[0], _B_VALUES[-1])),
            np.log10(np.clip(tex_arr.ravel(), _TEX_VALUES[0], _TEX_VALUES[-1])),
            np.log10(np.clip(ratio_arr.ravel(), _C12_C13_VALUES[0], _C12_C13_VALUES[-1])),
            np.log10(np.clip(nh2_arr.ravel(), *self.nh2_bounds)),
            np.log10(np.clip(n12co_arr.ravel(), *self.nco_bounds)),
        ))
        result = 10.0 ** self._interpolators[species](points)
        result = np.clip(result.reshape(n12co_arr.shape), 0.0, 1.0)
        return float(result) if result.ndim == 0 else result


@lru_cache(maxsize=1)
def _grid():
    return VisserCOShielding()


def fShield_CO_Visser(n12co, nh2, tex=20.0, b_co=0.3, c12_c13=60.0):
    return _grid().factor("12C16O", n12co, nh2, b_co=b_co, tex=tex, c12_c13=c12_c13)


def fShield_13CO_Visser(n12co, nh2, tex=20.0, b_co=0.3, c12_c13=60.0):
    return _grid().factor("13C16O", n12co, nh2, b_co=b_co, tex=tex, c12_c13=c12_c13)


def fShield_C18O_Visser(n12co, nh2, tex=20.0, b_co=0.3, c12_c13=60.0):
    return _grid().factor("12C18O", n12co, nh2, b_co=b_co, tex=tex, c12_c13=c12_c13)


def visser_table_metadata():
    return {
        "source": "Visser, van Dishoeck & Black 2009, A&A 503, 323",
        "b_grid_km_s": _B_VALUES.copy(),
        "tex_grid_K": _TEX_VALUES.copy(),
        "c12_c13_grid": _C12_C13_VALUES.copy(),
        "o16_o18_table_ratio": _O16_O18_TABLE_RATIO,
    }
