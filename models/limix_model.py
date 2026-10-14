import importlib
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

from models.base_model import BaseTabularPFM, Readout


def limix_root():
    """The folder holding LimiX's packages (`inference`, `model`, `utils`, `config`).

    It is $LIMIX_ROOT (a clone of https://github.com/limix-ldm-ai/LimiX, which then does not need
    to be installed), else where the installed `inference` package lives.
    """
    root = os.environ.get("LIMIX_ROOT")
    if root:
        if not os.path.isdir(os.path.join(root, "inference")):
            raise ImportError(f"$LIMIX_ROOT={root} is not a LimiX clone: it has no `inference` folder")
        return root
    spec = importlib.util.find_spec("inference")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "LimiX is not installed: clone https://github.com/limix-ldm-ai/LimiX and run "
            "`pip install --no-deps -e .` in it (see the readme), or set $LIMIX_ROOT to the clone"
        )
    return os.path.dirname(list(spec.submodule_search_locations)[0])


def import_limix_predictor():
    """Import LimiX's predictor next to this project's own `utils` package.

    LimiX installs a top-level package named `utils` (its predictor does
    `from utils.loading import ...`), as this project does. Python can hold only one, so LimiX's
    folder is added to the search path of the project's package: `utils.loading` and
    `utils.categorical_encoding` are then found in LimiX's folder, `utils.metrics` and
    `utils.plots` in ours. That works as long as no module has the same name in both.

    Returns:
        type: LimiX's `LimiXPredictor` factory (inference.predictor).
    """
    root = limix_root()
    if root not in sys.path:
        sys.path.append(root)  # after the project, so that its own packages keep priority
    import utils

    limix_utils = os.path.join(root, "utils")
    if limix_utils not in utils.__path__:
        modules = lambda path: {f[:-3] for f in os.listdir(path) if f.endswith(".py")} - {"__init__"}
        clash = modules(limix_utils) & modules(utils.__path__[0])
        if clash:
            raise ImportError(f"LimiX and this project both have utils modules named {sorted(clash)}")
        utils.__path__.append(limix_utils)

    from inference.predictor import LimiXPredictor

    # The v2 predictor creates a cache folder under a hard-coded /mnt/public/... path as soon as
    # it is built, which does not exist (or is not writable) on a cluster. The cache is only used
    # with use_data_cache=True, so it is moved to a temporary folder.
    v2 = importlib.import_module("inference.v2_0.predictor").LimiXPredictor
    if not getattr(v2.CacheManager, "_relocated", False):
        original = v2.CacheManager

        class RelocatedCacheManager(original):
            _relocated = True

            def __init__(self, cache_dir=None):
                super().__init__(cache_dir=os.environ.get("LIMIX_CACHE_DIR") or os.path.join(tempfile.gettempdir(), "limix_cache"))

        v2.CacheManager = RelocatedCacheManager
    return LimiXPredictor


def default_inference_config(n_estimators):
    """LimiX's default classification config (no retrieval), with only its first `n_estimators`
    preprocessing pipelines. Each pipeline is one member of the ensemble, with its own feature
    transform, column order and class permutation; one member keeps a single deterministic
    forward pass, as `n_estimators=1` does for the other models."""
    path = os.path.join(limix_root(), "config", "cls_default_noretrieval_v2.json")
    with open(path) as file:
        config = json.load(file)
    if not 1 <= n_estimators <= len(config["pipelines"]):
        raise ValueError(f"n_estimators must be between 1 and {len(config['pipelines'])} for LimiX")
    config["pipelines"] = config["pipelines"][:n_estimators]
    return config


class LimiXClassifier:
    """A scikit-learn-like face on LimiX's predictor, which has no `fit`: the training table is
    only kept, and given to `predict` together with the query rows each time, as LimiX does."""

    def __init__(self, predictor):
        self.predictor = predictor
        self.X = self.y = None
        self.classes_ = None

    def fit(self, X, y):
        self.X, self.y = np.asarray(X, dtype=np.float64), np.asarray(y)
        self.classes_ = np.unique(self.y)
        return self

    def predict_proba(self, X):
        return self.predictor.predict(self.X, self.y, np.asarray(X, dtype=np.float64), task_type="Classification")

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    @property
    def model(self):
        """The torch model, loaded when the predictor first needs it."""
        self.predictor._ensure_model_loaded()
        return self.predictor.model


class LimiX(BaseTabularPFM):
    """Shared LimiX wrapper (stable-ai), read with forward hooks. The versions live in
    limix_v2_model.py.

    LimiX-2 is one stack of 24 blocks, each attending across the features and across the rows.
    Every row is a sequence of feature tokens followed by 4 trailing target tokens (`y` tokens),
    and the decoder reads the target tokens of the query rows after the last block and a final
    norm. So the embedding of a query row is made of those tokens, and the positions are:

    - `--layer <i>` (0 to 23): the output of block `i`, the 4 target tokens of the query rows
      concatenated (4 x 256 = 1024 values per row).
    - `last` (the default): what the decoder reads, block 23's target tokens after the final
      norm, same shape.

    `--feature_slot` has no effect: the feature tokens are not read, the target tokens are where
    LimiX builds its prediction. `n_estimators` is the number of preprocessing pipelines of the
    ensemble (at most the 32 of LimiX's default config), 1 keeps a single forward pass; with
    more, the members' embeddings are averaged. `dict_params` go to LimiXPredictor, e.g.
    mix_precision or softmax_temperature.
    """

    LAYER_CONTAINERS = ("transformer_encoder.layers",)
    NORM_MODULE = "cls_y_encoder_out_norm"

    def _build_classifier(self):
        import torch

        predictor = import_limix_predictor()(
            device=torch.device(self.device),
            model_path=self.model_path,
            inference_config=default_inference_config(self.n_estimators),
            seed=self.seed,
            **self.dict_params,
        )
        return LimiXClassifier(predictor)

    def _torch_model(self):
        return self.clf.model

    def default_layers(self):
        return self.stack_layers() + ["last"]

    def _named_readouts(self):
        return {"last": Readout(self.NORM_MODULE, self._members_mean, select=lambda output, _kwargs: output[0])}

    def _module_readout(self, name):
        if not name.startswith("transformer_encoder.layers.") or not name.rsplit(".", 1)[1].isdigit():
            raise ValueError(f"LimiX can be read after a block ('<i>') or at 'last', not at '{name}'")
        model = self._torch_model()
        n_tokens = getattr(model, "y_token_k", None)
        if n_tokens is None:  # the model as the predictor holds it, else the config it was built from
            n_tokens = int(getattr(self.clf.predictor, "model_config", {}).get("num_cls_tokens", 1))

        def select(output, kwargs):
            # (1, rows, tokens, dim): the train rows come first in the sequence, the y tokens last
            return output[0, kwargs["eval_pos"] :, -n_tokens:, :]

        return Readout(name, self._members_mean, select=select)

    @staticmethod
    def _members_mean(capture):
        """The (query rows, y tokens, dim) chunks of every pass, as (n_test, tokens * dim).

        Passes over chunks of the query rows come in order; with several members, the passes of
        one member after the other. So the rows are n_test per member, averaged over them.
        """
        import torch

        rows = torch.cat(capture.chunks)
        if rows.shape[0] % capture.n_test:
            raise RuntimeError(f"Captured {rows.shape[0]} rows, expected a multiple of {capture.n_test}")
        rows = rows.reshape(-1, capture.n_test, *rows.shape[1:]).mean(dim=0)
        return rows.flatten(start_dim=1)
