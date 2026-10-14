from models.limix_model import LimiX
from models.tabpfn_model import find_hf_checkpoint


class LimiXV2(LimiX):
    """LimiX-2 (400M parameters, 24 blocks), the checkpoint of stable-ai/LimiX-2."""

    # LIMIX2_CKPT overrides the path.
    DEFAULT_MODEL_PATH = find_hf_checkpoint("stable-ai/LimiX-2", "LimiX-2.ckpt", "LIMIX2_CKPT")
