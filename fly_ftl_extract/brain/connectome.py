"""Load the mushroom-body subgraph extracted from FlyWire FAFB v783.

The data ships inside the package (``fly_ftl_extract/data/mb_fafb783.npz``, built by
``scripts/build_connectome.py``).  There is no statistical fallback: without the file the
fly does not exist and :func:`load` raises :class:`ConnectomeMissingError` with instructions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np
import scipy.sparse as sp

DATA_PACKAGE = "fly_ftl_extract.data"
NPZ_NAME = "mb_fafb783.npz"
META_NAME = "meta.json"

MIN_KC = 1500
MIN_PN = 50
MIN_MBON = 20
EXPECTED_APL = 1


class ConnectomeMissingError(FileNotFoundError):
    """The connectome file is absent."""


class ConnectomeInvalidError(ValueError):
    """The connectome file exists but fails validation."""


@dataclass(frozen=True)
class Connectome:
    """One hemisphere of the mushroom body as a signed synaptic-count matrix.

    ``weights[pre, post] = sign(pre) * syn_count`` in CSR form, so that
    ``spikes @ weights`` gives the synaptic drive of every neuron.  ``weights`` holds no
    explicit zeros: edges whose presynaptic neuron is a DAN (sign 0, no current) are kept
    apart in ``dan_edges`` (synapse counts, same neuron indexing) for the dopamine phase.
    """

    weights: sp.csr_matrix
    syn_count: sp.csr_matrix
    dan_edges: sp.csr_matrix
    root_id: np.ndarray
    cell_class: np.ndarray
    cell_sub_class: np.ndarray
    cell_type: np.ndarray
    side: np.ndarray
    glomerulus: np.ndarray
    nt: np.ndarray
    sign: np.ndarray
    pn_idx: np.ndarray
    kc_idx: np.ndarray
    apl_idx: np.ndarray
    mbon_idx: np.ndarray
    dan_idx: np.ndarray
    sha256: str
    meta: dict[str, object]

    @property
    def n_neurons(self) -> int:
        """Number of neurons in the subgraph."""
        return int(self.weights.shape[0])

    @property
    def n_edges(self) -> int:
        """Number of edges in ``weights`` (``syn_count >= 5``, DAN-presynaptic excluded)."""
        return int(self.weights.nnz)

    @property
    def n_dan_edges(self) -> int:
        """Number of DAN-presynaptic edges kept in ``dan_edges``."""
        return int(self.dan_edges.nnz)

    @property
    def hemisphere(self) -> str:
        """``left`` or ``right``."""
        return str(self.side[0])


def data_path(name: str) -> Path:
    """Filesystem path of a packaged data file (may not exist)."""
    return Path(str(resources.files(DATA_PACKAGE).joinpath(name)))


def _validate(c: Connectome) -> None:
    problems: list[str] = []
    if len(c.kc_idx) < MIN_KC:
        problems.append(f"{len(c.kc_idx)} Kenyon cells, expected >= {MIN_KC}")
    if len(c.pn_idx) < MIN_PN:
        problems.append(f"{len(c.pn_idx)} projection neurons, expected >= {MIN_PN}")
    if len(c.apl_idx) != EXPECTED_APL:
        problems.append(f"{len(c.apl_idx)} APL neurons, expected exactly {EXPECTED_APL}")
    if len(c.mbon_idx) < MIN_MBON:
        problems.append(f"{len(c.mbon_idx)} MBONs, expected >= {MIN_MBON}")
    n = c.n_neurons
    if c.weights.shape != (n, n) or len(c.root_id) != n or len(c.sign) != n:
        problems.append("array sizes disagree with the matrix shape")
    if len(set(c.side.tolist())) != 1:
        problems.append("neurons from more than one hemisphere")
    if (c.weights.data == 0).any():
        problems.append("weights contain explicit zeros")
    if c.weights[c.dan_idx].nnz:
        problems.append("DAN-presynaptic edges found in weights (they belong to dan_edges)")
    dan_rows = np.unique(c.dan_edges.tocoo().row)
    if not np.isin(dan_rows, c.dan_idx).all():
        problems.append("dan_edges has a non-DAN presynaptic neuron")
    if c.meta.get("sha256_npz") not in (None, c.sha256):
        problems.append("meta.json sha256 does not match the npz file")
    if problems:
        raise ConnectomeInvalidError("; ".join(problems))


def load(path: Path | None = None) -> Connectome:
    """Read and validate the packaged connectome (or the given ``path``)."""
    npz_path = path if path is not None else data_path(NPZ_NAME)
    if not npz_path.exists():
        msg = (
            f"connectome file {npz_path} is missing. The fly is real or nothing: build it with\n"
            "  uv run python scripts/build_connectome.py\n"
            "after downloading the FlyWire release into .cache/flywire/ (the script prints the\n"
            "download instructions). A wheel from PyPI already contains the file."
        )
        raise ConnectomeMissingError(msg)
    sha256 = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    meta_path = npz_path.with_name(META_NAME)
    meta: dict[str, object] = (
        json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    )
    with np.load(npz_path, allow_pickle=False) as z:
        n = len(z["root_id"])
        weights = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(n, n))
        syn = sp.csr_matrix((z["syn_count"], z["indices"], z["indptr"]), shape=(n, n))
        dan = sp.csr_matrix((z["dan_syn_count"], (z["dan_pre"], z["dan_post"])), shape=(n, n))
        connectome = Connectome(
            weights=weights,
            syn_count=syn,
            dan_edges=dan,
            root_id=z["root_id"],
            cell_class=z["cell_class"],
            cell_sub_class=z["cell_sub_class"],
            cell_type=z["cell_type"],
            side=z["side"],
            glomerulus=z["glomerulus"],
            nt=z["nt"],
            sign=z["sign"],
            pn_idx=z["pn_idx"],
            kc_idx=z["kc_idx"],
            apl_idx=z["apl_idx"],
            mbon_idx=z["mbon_idx"],
            dan_idx=z["dan_idx"],
            sha256=sha256,
            meta=meta,
        )
    _validate(connectome)
    return connectome
