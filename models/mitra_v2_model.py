from models.mitra_model import Mitra


class MitraV2(Mitra):
    """Mitra v2 (arXiv 2609.04540): the architecture of v1, pre-trained on a scaled-up
    synthetic distribution (10x longer context, 3x more features, better optimizer)."""

    DEFAULT_MODEL_PATH = "autogluon/mitra-classifier-2"
