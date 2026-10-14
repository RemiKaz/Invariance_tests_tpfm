from models.tabpfn_model import TabPFN, find_hf_checkpoint


class TabPFNV2(TabPFN):
    """TabPFN v2."""

    # The transformer is a flat list of TabPFNBlock, addressed by `--layer <index>`.
    LAYER_CONTAINERS = ("blocks",) + TabPFN.LAYER_CONTAINERS
    # One flat stack, so the depth can be named: `--layer early` / `middle` / `last`.
    LAYER_FRACTIONS = {"early": 0.25, "middle": 0.5}
    # The library infers the model version from the file name, so this one is loaded as v2.
    # TABPFN_V2_CKPT overrides the path.
    DEFAULT_MODEL_PATH = find_hf_checkpoint(
        "Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt", "TABPFN_V2_CKPT"
    )
