"""The whole fly: tokenizer → odour → mushroom body → MBON readout → keys.

:class:`Judge` turns Python sources into the keys the fly smells in them.  Every
candidate with a key name is sniffed with its production seed
(:func:`fly_ftl_extract.dopamine.seed.trial_seed`): its window becomes a sequence of puffs
(``odor/encoder.py``), the mushroom body receives them one after another
(:meth:`Brain.simulate_sequence`) and the readout reads the Kenyon-cell counts of every
puff.  When the MBON margin is closer to zero than θ the fly sniffs again (``max_resniff``
extra trials, all of them) and the summed margin decides.  Keyword arguments are judged
the same way, only for calls judged to be keys.

Nothing here looks at names or shapes of the code: the decision is the sign of a margin
computed from Kenyon-cell spikes.

Cost model: one ``simulate_sequence`` call costs ~1 s for 5700 steps whatever the batch
(up to ~256 trials), so the judge gathers the windows of *many files* into four rounds —
keys, key resniffs, kwargs, kwarg resniffs — instead of four rounds per file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fly_ftl_extract.brain import DEFAULT_PARAMS, Brain, BrainParams, Connectome, SequenceResult
from fly_ftl_extract.brain import connectome as connectome_module
from fly_ftl_extract.dopamine import weights as weights_module
from fly_ftl_extract.dopamine.readout import Readout, vote
from fly_ftl_extract.dopamine.seed import kwarg_trial_index, salted_seed, sniff_seed, trial_seed
from fly_ftl_extract.dopamine.weights import MbonWeights
from fly_ftl_extract.ftl.model import ExtractOptions
from fly_ftl_extract.odor.encoder import ENCODER_VERSION, N_SLOTS, encode_many
from fly_ftl_extract.tokenizer.candidates import Candidate, Kwarg, Window, iter_candidates

BATCH = 256
"""Trials per ``simulate_sequence`` call.  ``docs/bench_sequence.json``: in one process the
throughput grows with the batch (65 → 84 → 102 trials/s for 64 / 128 / 256); with many
processes memory bandwidth limits and 64 wins — ``cli/extract.py`` picks per mode."""

FileInput = tuple[list[Candidate], bytes]
"""What the judge needs of one file: its candidates and its raw bytes (seed material)."""


@dataclass(frozen=True)
class Verdict:
    """The fly's answer about one window."""

    is_positive: bool
    margin: float
    """Summed margin over all sniffs."""
    sniffs: int
    """Trials used (base trials + resniffs)."""
    kc_active_fraction: float
    """Of the first sniff, averaged over the puffs that carried odour (for the TUI)."""
    pn_active_fraction: float = 0.0
    """Share of projection neurons that spiked at least once in the first sniff."""
    apl_spikes: int = 0
    """APL spikes over the whole first sniff."""
    mbon_active_fraction: float = 0.0
    """Share of MBONs that spiked at least once in the first sniff."""
    kc_pattern: bytes = b""
    """``np.packbits`` of the Kenyon cells that spiked in any puff of the first sniff
    (``n_kc`` bits) — the row of the TUI spike raster.  Real spikes, not decoration."""

    def kc_bits(self, n_kc: int) -> np.ndarray:
        """The ``kc_pattern`` unpacked to ``n_kc`` booleans (all ``False`` if not recorded)."""
        if not self.kc_pattern:
            return np.zeros(n_kc, dtype=bool)
        return np.unpackbits(np.frombuffer(self.kc_pattern, dtype=np.uint8))[:n_kc].astype(bool)


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

    def all_verdicts(self) -> list[Verdict]:
        """Every verdict of the file: candidates first, then the keys' keyword arguments."""
        out = [v for v in self.verdicts if v is not None]
        out.extend(v for k in self.keys for _, v in k.kwarg_verdicts)
        return out

    @property
    def trials(self) -> int:
        """Brain trials spent on this file (candidates and keyword arguments)."""
        return sum(v.sniffs for v in self.all_verdicts())

    def resniffs(self, base_trials: int) -> int:
        """Extra sniffs beyond the ``base_trials`` every window gets."""
        return sum(v.sniffs - base_trials for v in self.all_verdicts())


class Judge:
    """Loads the brain and the weights once, then judges files.

    Args:
        options: the run's options (they drive the token normalisation, not the verdict).
        connectome: an already loaded connectome (loaded from the package otherwise).
        params: LIF parameters; must be the ones the weights were trained with.
        weights: trained readouts (loaded from the package otherwise, hashes checked).
        max_resniff: extra sniffs taken when ``|margin| < θ`` (default: from the weights).
        base_trials: ``--fly-trials`` — sniffs every window gets before the resniff rule
            (their margins are summed; θ is compared with the sum).
        seed_salt: ``--fly-seed`` — mixed into every trial seed (0 = production seeds).
        batch: trials per brain call.
    """

    def __init__(
        self,
        options: ExtractOptions,
        *,
        connectome: Connectome | None = None,
        params: BrainParams = DEFAULT_PARAMS,
        weights: MbonWeights | None = None,
        max_resniff: int | None = None,
        base_trials: int = 1,
        seed_salt: int = 0,
        batch: int = BATCH,
    ) -> None:
        if base_trials < 1:
            msg = "base_trials must be at least 1"
            raise ValueError(msg)
        if batch < 1:
            msg = "batch must be at least 1"
            raise ValueError(msg)
        self.options = options
        cx = connectome if connectome is not None else connectome_module.load()
        self.brain = Brain(cx, params)
        self.weights = (
            weights
            if weights is not None
            else weights_module.load(brain_hash=self.brain.hash, encoder_version=ENCODER_VERSION)
        )
        self.max_resniff = self.weights.max_resniff if max_resniff is None else max_resniff
        self.base_trials = base_trials
        self.seed_salt = seed_salt
        self.batch = batch
        self.n_kc = self.brain.n_kc
        expected = N_SLOTS * self.n_kc
        for name, readout in (("key", self.weights.key), ("kwarg", self.weights.kwarg)):
            if readout.n_states != expected:
                msg = (
                    f"{name} readout expects {readout.n_states} KC states, the fly has "
                    f"{N_SLOTS} puffs × {self.n_kc} KC = {expected}. Retrain."
                )
                raise weights_module.WeightsMismatchError(msg)

    # ------------------------------------------------------------------------ sniffing

    def sniff(
        self, windows: list[Window], seeds: np.ndarray, readout: Readout, theta: float
    ) -> list[Verdict]:
        """Judge windows: ``base_trials`` each, then ``max_resniff`` more where ``|Σ| < θ``.

        Two rounds of brain calls: all base trials, then all extra trials of the unsure
        windows.  Every trial has its own seed, so how trials are grouped into batches
        does not change a single spike.
        """
        if not windows:
            return []
        odors = encode_many(windows, self.options)
        n = len(windows)
        base = self.base_trials
        margins = np.zeros((n, base + self.max_resniff), dtype=np.float32)
        first: list[Verdict | None] = [None] * n
        self._run_trials(
            odors,
            seeds,
            np.repeat(np.arange(n), base),
            np.tile(np.arange(base), n),
            readout,
            margins,
            first,
        )
        unsure = np.flatnonzero(np.abs(margins[:, :base].sum(axis=1)) < theta)
        if len(unsure) and self.max_resniff:
            self._run_trials(
                odors,
                seeds,
                np.repeat(unsure, self.max_resniff),
                np.tile(np.arange(base, base + self.max_resniff), len(unsure)),
                readout,
                margins,
                None,
            )
        sniffs = np.full(n, base, dtype=np.int32)
        sniffs[unsure] = base + self.max_resniff
        summed = vote(margins)
        out: list[Verdict] = []
        for i in range(n):
            stats = first[i]
            assert stats is not None  # noqa: S101 — every window had a first sniff
            out.append(
                Verdict(
                    is_positive=bool(summed[i] > 0),
                    margin=float(summed[i]),
                    sniffs=int(sniffs[i]),
                    kc_active_fraction=stats.kc_active_fraction,
                    pn_active_fraction=stats.pn_active_fraction,
                    apl_spikes=stats.apl_spikes,
                    mbon_active_fraction=stats.mbon_active_fraction,
                    kc_pattern=stats.kc_pattern,
                )
            )
        return out

    def _run_trials(  # noqa: PLR0917
        self,
        odors: np.ndarray,
        seeds: np.ndarray,
        idx: np.ndarray,
        ks: np.ndarray,
        readout: Readout,
        margins: np.ndarray,
        first: list[Verdict | None] | None,
    ) -> None:
        """Trial ``(idx[j], ks[j])`` = window ``idx[j]``, sniff ``ks[j]``; batched brain calls."""
        trial_seeds = np.array(
            [sniff_seed(int(seeds[i]), int(k)) for i, k in zip(idx, ks, strict=True)],
            dtype=np.uint64,
        )
        for start in range(0, len(idx), self.batch):
            sl = slice(start, start + self.batch)
            res = self.brain.simulate_sequence(odors[idx[sl]], trial_seeds[sl])
            margins[idx[sl], ks[sl]] = readout.margin(res.kc_counts)
            if first is not None:
                stats = self._first_sniff_stats(res)
                for j, (i, k) in enumerate(zip(idx[sl], ks[sl], strict=True)):
                    if k == 0:
                        first[int(i)] = stats[j]

    @staticmethod
    def _first_sniff_stats(res: SequenceResult) -> list[Verdict]:
        """What the TUI shows about a trial: real activity of PN, KC, APL and MBON."""
        fractions = _active_over_puffs(res.kc_active_fraction, res.puff_active)
        kc_any = res.kc_counts.sum(axis=1) > 0  # (n_trials, n_kc)
        pn_any = (res.pn_counts.sum(axis=1) > 0).mean(axis=1)
        mbon_any = (res.mbon_counts.sum(axis=1) > 0).mean(axis=1)
        apl = res.apl_counts.sum(axis=1)
        return [
            Verdict(
                is_positive=False,
                margin=0.0,
                sniffs=1,
                kc_active_fraction=float(fractions[i]),
                pn_active_fraction=float(pn_any[i]),
                apl_spikes=int(apl[i]),
                mbon_active_fraction=float(mbon_any[i]),
                kc_pattern=np.packbits(kc_any[i]).tobytes(),
            )
            for i in range(res.n_trials)
        ]

    # ------------------------------------------------------------------------- judging

    def _seed(self, content: bytes, index: int) -> int:
        return salted_seed(trial_seed(content, index, ENCODER_VERSION), self.seed_salt)

    def judge_source(self, source: str, content: bytes) -> JudgedFile:
        """Keys of one file (``content`` = the raw bytes, part of every trial seed)."""
        return self.judge_candidates(list(iter_candidates(source, self.options)), content)

    def judge_candidates(self, candidates: list[Candidate], content: bytes) -> JudgedFile:
        """Keys among already tokenized ``candidates`` of the file whose bytes are ``content``."""
        return self.judge_many([(candidates, content)])[0]

    def judge_many(self, files: list[FileInput]) -> list[JudgedFile]:
        """Judge several files in four brain rounds (keys, resniffs, kwargs, resniffs)."""
        judged = [JudgedFile(candidates, [None] * len(candidates)) for candidates, _ in files]
        refs = [
            (f, i)
            for f, (candidates, _) in enumerate(files)
            for i, c in enumerate(candidates)
            if c.key_name is not None
        ]
        seeds = np.array([self._seed(files[f][1], i) for f, i in refs], dtype=np.uint64)
        verdicts = self.sniff(
            [files[f][0][i].window for f, i in refs],
            seeds,
            self.weights.key,
            self.weights.theta_key,
        )
        for (f, i), verdict in zip(refs, verdicts, strict=True):
            judged[f].verdicts[i] = verdict
        positives = [(f, i) for (f, i), v in zip(refs, verdicts, strict=True) if v.is_positive]
        # keyword arguments of the keys, judged in one round
        kwarg_refs = [(f, i, k) for f, i in positives for k in range(len(files[f][0][i].kwargs))]
        kwarg_seeds = np.array(
            [self._seed(files[f][1], kwarg_trial_index(i, k)) for f, i, k in kwarg_refs],
            dtype=np.uint64,
        )
        kwarg_verdicts = self.sniff(
            [files[f][0][i].kwargs[k].window for f, i, k in kwarg_refs],
            kwarg_seeds,
            self.weights.kwarg,
            self.weights.theta_kwarg,
        )
        by_candidate: dict[tuple[int, int], list[tuple[Kwarg, Verdict]]] = {
            ref: [] for ref in positives
        }
        for (f, i, k), v in zip(kwarg_refs, kwarg_verdicts, strict=True):
            by_candidate[(f, i)].append((files[f][0][i].kwargs[k], v))
        for f, i in positives:
            c = files[f][0][i]
            assert c.key_name is not None  # noqa: S101 — nameable by construction
            key_verdict = judged[f].verdicts[i]
            assert key_verdict is not None  # noqa: S101 — positives were judged
            pairs = tuple(by_candidate[(f, i)])
            judged[f].keys.append(
                JudgedKey(
                    candidate=c,
                    key_name=c.key_name,
                    verdict=key_verdict,
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
