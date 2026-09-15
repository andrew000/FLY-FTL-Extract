"""The whole fly: tokenizer → odour → mushroom body → MBON readout → keys.

:class:`Judge` turns one Python source into the keys the fly smells in it.  Every
candidate with a key name is sniffed once with its production seed
(:func:`fly_ftl_extract.dopamine.seed.trial_seed`): its window becomes a sequence of puffs
(``odor/encoder.py``), the mushroom body receives them one after another
(:meth:`Brain.simulate_sequence`) and the readout reads the Kenyon-cell counts of every
puff.  When the MBON margin is closer to zero than θ the fly sniffs again (up to
``max_resniff`` extra trials) and the summed margin decides.  Keyword arguments are judged
the same way, only for calls judged to be keys.

Nothing here looks at names or shapes of the code: the decision is the sign of a margin
computed from Kenyon-cell spikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, BrainParams, Connectome
from fly_ftl_extract.brain import connectome as connectome_module
from fly_ftl_extract.dopamine import weights as weights_module
from fly_ftl_extract.dopamine.readout import Readout, vote
from fly_ftl_extract.dopamine.seed import kwarg_trial_index, sniff_seed, trial_seed
from fly_ftl_extract.dopamine.weights import MbonWeights
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.odor.encoder import ENCODER_VERSION, N_SLOTS, encode_many
from fly_ftl_extract.tokenizer.candidates import Candidate, Kwarg, Window, iter_candidates

BATCH = 256


@dataclass(frozen=True)
class Verdict:
    """The fly's answer about one window."""

    is_positive: bool
    margin: float
    """Summed margin over all sniffs."""
    sniffs: int
    """Trials used (1 + resniffs)."""
    kc_active_fraction: float
    """Of the first sniff, averaged over the puffs that carried odour (for the TUI)."""


@dataclass(frozen=True)
class JudgedKey:
    """A candidate the fly called a key, with its placeable keyword arguments."""

    candidate: Candidate
    key_name: str
    verdict: Verdict
    placeable: tuple[str, ...]
    kwarg_verdicts: tuple[tuple[Kwarg, Verdict], ...]

    @property
    def call_position(self) -> tuple[int, int]:
        """``(line, column)`` of the call, as the original reports it."""
        pos = self.candidate.call_position
        assert pos is not None  # noqa: S101 — a key always belongs to a call
        return pos

    @property
    def path_value(self) -> str | None:
        """The ``_path=`` literal of the call, if any."""
        return next((k.path_value for k in self.candidate.kwargs if k.name == "_path"), None)


@dataclass
class JudgedFile:
    """Every candidate's verdict and the keys of one file."""

    candidates: list[Candidate]
    verdicts: list[Verdict | None]
    """``None`` for candidates that cannot name a key (chain of a get-like call)."""
    keys: list[JudgedKey] = field(default_factory=list)


class Judge:
    """Loads the brain and the weights once, then judges files."""

    def __init__(
        self,
        options: ExtractOptions,
        *,
        connectome: Connectome | None = None,
        params: BrainParams = DEFAULT_PARAMS,
        weights: MbonWeights | None = None,
        max_resniff: int | None = None,
    ) -> None:
        self.options = options
        cx = connectome if connectome is not None else connectome_module.load()
        self.brain = Brain(cx, params)
        self.weights = (
            weights
            if weights is not None
            else weights_module.load(brain_hash=self.brain.hash, encoder_version=ENCODER_VERSION)
        )
        self.max_resniff = self.weights.max_resniff if max_resniff is None else max_resniff
        self.n_kc = self.brain.n_kc
        expected = N_SLOTS * self.n_kc
        for name, readout in (("key", self.weights.key), ("kwarg", self.weights.kwarg)):
            if readout.n_states != expected:
                msg = (
                    f"{name} readout expects {readout.n_states} KC states, the fly has "
                    f"{N_SLOTS} puffs × {self.n_kc} KC = {expected}. Retrain."
                )
                raise weights_module.WeightsMismatchError(msg)

    def sniff(
        self, windows: list[Window], seeds: np.ndarray, readout: Readout, theta: float
    ) -> list[Verdict]:
        """Judge windows: one trial each, then extra sniffs where ``|margin| < θ``."""
        if not windows:
            return []
        odors = encode_many(windows, self.options)
        margins = np.zeros((len(windows), 1 + self.max_resniff), dtype=np.float32)
        fractions = np.zeros(len(windows), dtype=np.float32)
        sniffs = np.ones(len(windows), dtype=np.int32)
        for start in range(0, len(windows), BATCH):
            sl = slice(start, start + BATCH)
            res = self.brain.simulate_sequence(odors[sl], seeds[sl])
            margins[sl, 0] = readout.margin(res.kc_counts)
            fractions[sl] = _active_over_puffs(res.kc_active_fraction, res.puff_active)
        unsure = np.flatnonzero(np.abs(margins[:, 0]) < theta)
        for sniff in range(1, self.max_resniff + 1):
            if len(unsure) == 0:
                break
            extra_seeds = np.array(
                [sniff_seed(int(s), sniff) for s in seeds[unsure]], dtype=np.uint64
            )
            for start in range(0, len(unsure), BATCH):
                idx = unsure[start : start + BATCH]
                res = self.brain.simulate_sequence(odors[idx], extra_seeds[start : start + BATCH])
                margins[idx, sniff] = readout.margin(res.kc_counts)
            sniffs[unsure] = sniff + 1
        summed = vote(margins)
        return [
            Verdict(bool(summed[i] > 0), float(summed[i]), int(sniffs[i]), float(fractions[i]))
            for i in range(len(windows))
        ]

    def judge_source(self, source: str, content: bytes) -> JudgedFile:
        """Keys of one file (``content`` = the raw bytes, part of every trial seed)."""
        candidates = list(iter_candidates(source, self.options))
        judged = JudgedFile(candidates, [None] * len(candidates))
        nameable = [i for i, c in enumerate(candidates) if c.key_name is not None]
        seeds = np.array(
            [trial_seed(content, i, ENCODER_VERSION) for i in nameable], dtype=np.uint64
        )
        verdicts = self.sniff(
            [candidates[i].window for i in nameable],
            seeds,
            self.weights.key,
            self.weights.theta_key,
        )
        for i, verdict in zip(nameable, verdicts, strict=True):
            judged.verdicts[i] = verdict
        positives = [i for i, v in zip(nameable, verdicts, strict=True) if v.is_positive]
        # keyword arguments of the keys, judged in one batch
        kwarg_refs: list[tuple[int, int]] = [
            (i, k) for i in positives for k in range(len(candidates[i].kwargs))
        ]
        kwarg_seeds = np.array(
            [trial_seed(content, kwarg_trial_index(i, k), ENCODER_VERSION) for i, k in kwarg_refs],
            dtype=np.uint64,
        )
        kwarg_verdicts = self.sniff(
            [candidates[i].kwargs[k].window for i, k in kwarg_refs],
            kwarg_seeds,
            self.weights.kwarg,
            self.weights.theta_kwarg,
        )
        by_candidate: dict[int, list[tuple[Kwarg, Verdict]]] = {i: [] for i in positives}
        for (i, k), v in zip(kwarg_refs, kwarg_verdicts, strict=True):
            by_candidate[i].append((candidates[i].kwargs[k], v))
        for i in positives:
            c = candidates[i]
            assert c.key_name is not None  # noqa: S101 — nameable by construction
            pairs = tuple(by_candidate[i])
            judged.keys.append(
                JudgedKey(
                    candidate=c,
                    key_name=c.key_name,
                    verdict=judged.verdicts[i]
                    or Verdict(is_positive=True, margin=0.0, sniffs=0, kc_active_fraction=0.0),
                    placeable=tuple(k.name for k, v in pairs if v.is_positive),
                    kwarg_verdicts=pairs,
                )
            )
        return judged


def _active_over_puffs(fraction: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Mean active-KC share per trial over the puffs that carried odour (0 if none)."""
    n = active.sum(axis=1)
    return np.where(n > 0, (fraction * active).sum(axis=1) / np.maximum(n, 1), 0.0).astype(
        np.float32
    )
