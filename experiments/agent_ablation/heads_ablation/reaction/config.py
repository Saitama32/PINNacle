"""Configuration shared by the reaction-diffusion offline diagnostics.

Keep experiment identity and action-space identity together. Silently using an
incompatible action dictionary for a checkpoint would invalidate every
head-level conclusion. The G/B agents analyzed here both used five optimizers.
"""

from collections import OrderedDict


WORKSPACE = "saitama32"
TRAINING_PROJECT = (
    "rlpinn-reaction-diffusion-nu0-01-rho20-0-optimization-expand-set"
)
FARM_PROJECT = "rlpinn-reaction-diffusion-nu0-01-rho20-0-tolerance-expand-set"


REACTION_LEGACY_3 = OrderedDict(
    {
        "Adam": {"lr": [1e-2, 1e-3, 1e-4], "epochs": [100, 1000, 2500]},
        "LBFGS": {"lr": [1.0, 5e-1, 1e-1], "epochs": [100, 500, 1000]},
        "PSO": {"lr": [0.0, 1e-3, 1e-4], "epochs": [100, 200, 300]},
    }
)

REACTION_CURRENT_5 = OrderedDict(
    {
        **REACTION_LEGACY_3,
        "SOAP": {"lr": [1e-2, 1e-3, 3e-4], "epochs": [100, 1000, 2500]},
        "Muon": {"lr": [2e-2, 1e-2, 5e-3], "epochs": [100, 1000, 2500]},
    }
)

ACTION_SPACES = {
    "reaction_legacy_3": REACTION_LEGACY_3,
    "reaction_current_5": REACTION_CURRENT_5,
}

# First forensic comparison. G/B and the current reaction-diffusion agent use
# the same five-optimizer action space.
EXPERIMENTS = {
    "G": {
        "key": "037a7084956e4c2bae7ec477bb48d2cf",
        "project": TRAINING_PROJECT,
        "action_space": "reaction_current_5",
    },
    "B": {
        "key": "f1bec88bf5974a86b46dd13c503e1a36",
        "project": TRAINING_PROJECT,
        "action_space": "reaction_current_5",
    },
}

# Four independent runs for every parameter-target ablation.  These keys are
# kept as a named cohort so the return-transfer analysis is reproducible and
# does not depend on an analyst's local CSV directory.
TARGET_ABLATION_EXPERIMENTS = {
    **EXPERIMENTS,
    **{
        f"PPSW_N_{index + 1}": {
            "key": key,
            "project": TRAINING_PROJECT,
            "action_space": "reaction_current_5",
        }
        for index, key in enumerate(
            (
                "a978c89cf0464233b8951a729e90eb4f",
                "783197f3556d45398c70d394f80070b5",
                "d931378a0671492fb27a64638b312b7d",
                "7d479bb9cd5c4dc8946eee17c9ae19a7",
            )
        )
    },
    **{
        f"JOINT_SW_N_{index + 1}": {
            "key": key,
            "project": TRAINING_PROJECT,
            "action_space": "reaction_current_5",
        }
        for index, key in enumerate(
            (
                "23d205aa4ec743259cf400c2297d1e47",
                "0012d8d57acb4636b877e78f5eb14c92",
                "4bab178c95a945e08d2593b00b0a0a43",
                "8654f91328474a879c842a2c5d876d9c",
            )
        )
    },
    **{
        f"GLOBAL_1STEP_N_{index + 1}": {
            "key": key,
            "project": TRAINING_PROJECT,
            "action_space": "reaction_current_5",
        }
        for index, key in enumerate(
            (
                "2407220a9ed24f76ae54b9383101b617",
                "5319d9bb678741ca96930d1ee6dfa88b",
                "6f404d097a584539a32661af167f4e2d",
                "6b6eb1374a1f4e8dbc6e915c6cd7edfd",
            )
        )
    },
    **{
        f"LEGACY_N_{index + 1}": {
            "key": key,
            "project": TRAINING_PROJECT,
            "action_space": "reaction_current_5",
        }
        for index, key in enumerate(
            (
                "f56775e504ac4cd9a4f83634819f9189",
                "9b18236955d04a6f829eff2c541790e8",
                "1d6e304ced7b4413bc56f98db73507e2",
                "ddfc1dff19db47cfb6e021d6f695d754",
            )
        )
    },
}

# Earlier runs that isolate parameter targets without contribution
# normalization.  The first four global-one-step runs below are the standard
# five-optimizer agent; the separate "with SOAP optimizer" DQN-SOAP block is
# intentionally excluded from this clean credit-assignment cohort.
UNNORMALIZED_TARGET_EXPERIMENTS = {
    **{
        f"GLOBAL_1STEP_U_{index + 1}": {
            "key": key,
            "project": TRAINING_PROJECT,
            "action_space": "reaction_current_5",
        }
        for index, key in enumerate(
            (
                "93af321143f5432dbdf953788c3e7b9a",
                "6fe1c5ad480443098bde98e40df3bcf8",
                "c39a13908b6244e2b8e1123be6042aca",
                "cdae3dc015c04359a96507588895b369",
            )
        )
    },
    **{
        f"PPSW_U_{index + 1}": {
            "key": key,
            "project": TRAINING_PROJECT,
            "action_space": "reaction_current_5",
        }
        for index, key in enumerate(
            (
                "c6af2a222fb24744b6103a23c6a5994b",
                "96067f30c5054231bcc05970489bffe4",
                "444bf8c164e14313bd3ad77b819ff273",
                "6f5cc25711524ed5abf3acaef0bf3e37",
            )
        )
    },
}

CREDIT_SPECIALIZATION_EXPERIMENTS = {
    **EXPERIMENTS,
    **UNNORMALIZED_TARGET_EXPERIMENTS,
    **{
        label: config
        for label, config in TARGET_ABLATION_EXPERIMENTS.items()
        if label.startswith("LEGACY_N_")
    },
}

EXPERIMENT_COHORTS = {
    "gb": EXPERIMENTS,
    "target_ablation": TARGET_ABLATION_EXPERIMENTS,
    "credit_specialization": CREDIT_SPECIALIZATION_EXPERIMENTS,
}

DEFAULT_GAMMA = 0.9
DEFAULT_FARM_EXPERIMENTS = 8
