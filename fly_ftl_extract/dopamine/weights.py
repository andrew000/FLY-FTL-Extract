"""``data/mbon_weights.npz``: the trained readouts with the hashes they are valid for."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from fly_ftl_extract.brain.connectome import data_path
from fly_ftl_extract.dopamine.readout import FEATURE_MODES, FeatureMode, Readout

WEIGHTS_NAME = "mbon_weights.npz"


class WeightsMissingError(FileNotFoundError):
    """No trained readout: run ``scripts/make_dataset.py`` and ``scripts/train.py``."""


class WeightsMismatchError(ValueError):
    """The weights were trained for another brain or encoder version."""


@dataclass(frozen=True)
class MbonWeights:
    """Both readouts plus the provenance that makes them valid."""

    key: Readout
    kwarg: Readout
    theta_key: float
    theta_kwarg: float
    brain_hash: str
    encoder_version: str
    metrics: dict[str, object]
    max_resniff: int = 3

    def save(self, path: Path) -> None:
        """Write ``mbon_weights.npz`` (no pickles)."""
        np.savez(
            path,
            W_key=self.key.w.astype(np.float32),
            b_key=np.float32(self.key.b),
            W_kw=self.kwarg.w.astype(np.float32),
            b_kw=np.float32(self.kwarg.b),
            theta=np.array([self.theta_key, self.theta_kwarg], dtype=np.float32),
            brain_hash=np.array(self.brain_hash),
            encoder_version=np.array(self.encoder_version),
            feature_mode=np.array([self.key.mode, self.kwarg.mode]),
            max_resniff=np.int32(self.max_resniff),
            metrics=np.array(json.dumps(self.metrics)),
        )


def weights_path() -> Path:
    """Where the packaged weights live (may not exist)."""
    return data_path(WEIGHTS_NAME)


def load(
    path: Path | None = None, *, brain_hash: str | None = None, encoder_version: str | None = None
) -> MbonWeights:
    """Read the weights; raise :class:`WeightsMismatchError` when the hashes differ."""
    npz_path = path if path is not None else weights_path()
    if not npz_path.exists():
        msg = (
            f"{npz_path} is missing: the fly has not been trained. Run\n"
            "  uv run python scripts/make_dataset.py && uv run python scripts/train.py"
        )
        raise WeightsMissingError(msg)
    with np.load(npz_path, allow_pickle=False) as z:
        modes = [str(m) for m in z["feature_mode"]]
        if any(m not in FEATURE_MODES for m in modes):
            msg = f"unknown feature mode in {npz_path}: {modes}"
            raise WeightsMismatchError(msg)
        weights = MbonWeights(
            key=Readout(
                z["W_key"].astype(np.float32), float(z["b_key"]), cast("FeatureMode", modes[0])
            ),
            kwarg=Readout(
                z["W_kw"].astype(np.float32), float(z["b_kw"]), cast("FeatureMode", modes[1])
            ),
            theta_key=float(z["theta"][0]),
            theta_kwarg=float(z["theta"][1]),
            brain_hash=str(z["brain_hash"]),
            encoder_version=str(z["encoder_version"]),
            metrics=json.loads(str(z["metrics"])),
            max_resniff=int(z["max_resniff"]),
        )
    problems = []
    if brain_hash is not None and weights.brain_hash != brain_hash:
        problems.append(f"brain hash {weights.brain_hash} != {brain_hash}")
    if encoder_version is not None and weights.encoder_version != encoder_version:
        problems.append(f"encoder version {weights.encoder_version!r} != {encoder_version!r}")
    if problems:
        msg = f"{npz_path} was trained for another fly: " + "; ".join(problems) + ". Retrain."
        raise WeightsMismatchError(msg)
    return weights
