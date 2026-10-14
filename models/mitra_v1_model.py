from models.mitra_model import Mitra


class MitraV1(Mitra):
    """Mitra v1, the classifier of the Mitra paper (arXiv 2510.21204)."""

    DEFAULT_MODEL_PATH = "autogluon/mitra-classifier"
