"""Dopamine: training and use of the MBON readout (the fly's learned judgement)."""

from fly_ftl_extract.dopamine.judge import Judge, JudgedFile, JudgedKey, Verdict
from fly_ftl_extract.dopamine.readout import (
    FEATURE_MODES,
    FeatureMode,
    Readout,
    Scores,
    TrainConfig,
    dan_update,
    features,
    resniff_threshold,
    score,
    train_readout,
    vote,
)
from fly_ftl_extract.dopamine.seed import sniff_seed, trial_seed
from fly_ftl_extract.dopamine.weights import (
    MbonWeights,
    WeightsMismatchError,
    WeightsMissingError,
    weights_path,
)

__all__ = [
    "FEATURE_MODES",
    "FeatureMode",
    "Judge",
    "JudgedFile",
    "JudgedKey",
    "MbonWeights",
    "Readout",
    "Scores",
    "TrainConfig",
    "Verdict",
    "WeightsMismatchError",
    "WeightsMissingError",
    "dan_update",
    "features",
    "resniff_threshold",
    "score",
    "sniff_seed",
    "train_readout",
    "trial_seed",
    "vote",
    "weights_path",
]
