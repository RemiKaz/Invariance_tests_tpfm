import re

import numpy as np

from models.base_model import BaseTabularPFM, Readout


class Mitra(BaseTabularPFM):
    """Shared Mitra wrapper (Amazon, through AutoGluon), read with forward hooks.

    Mitra is a stack of 12 identical layers, each attending across the rows (the query
    rows attend to the support rows, column by column) and then across the features (row
    by row). It has no separate column and row stages, so the positions are numbered layers:

    - `--layer <i>` (0 to 11): the output of layer `i` for the query rows, shaped
      (rows, features + 1, dim). The first token of each row is the target token, the
      others are the features.
    - `early` and `middle`: the outputs of the layers that close the first quarter and the
      first half of the stack (layers 2 and 5 of 12).
    - `last` (the default, also `final_layer_norm`): the output of the last layer after its
      final norm, right before the head that predicts, same shape.

    `--feature_slot` reduces the tokens of a row to one vector: "target" (the default) keeps
    the target token, "mean" averages the feature tokens, "flatten" concatenates them.

    The classifier is used in-context, without fine-tuning. The versions live in
    mitra_v1_model.py and mitra_v2_model.py, their DEFAULT_MODEL_PATH is the Hugging Face repo
    of the checkpoint, read from the cache filled by down_weight.py.
    """

    LAYER_CONTAINERS = ("layers",)
    LAYER_FRACTIONS = {"early": 0.25, "middle": 0.5}
    FINAL_MODULE = "final_layer_norm"

    def _build_classifier(self):
        from autogluon.tabular.models.mitra._internal.models import tab2d
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier

        # Mitra uses flash-attention as soon as it is installed. It needs an Ampere GPU (the
        # V100 nodes fail with "FlashAttention only supports Ampere GPUs or newer"), and it
        # packs the tokens of all the rows in one axis, which loses the (rows, features + 1,
        # dim) layout the layers are read in. Tab2D reads this flag when it is built, so it
        # has to be off before the classifier is fitted.
        tab2d.FLASH_ATTN_AVAILABLE = False

        return MitraClassifier(
            device=self.device,
            fine_tune=False,
            n_estimators=self.n_estimators,
            hf_model=self.model_path,
            seed=self.seed,
            verbose=False,
            **self.dict_params,
        )

    def _torch_model(self):
        trainers = self.clf.trainers
        if not trainers:
            raise RuntimeError("The classifier has to be fitted before its layers can be read")
        return trainers[0].model

    def fit(self, X_train, y_train):
        # MitraClassifier holds out 20% of the rows to validate a fine-tuning that does not
        # happen here, and fits its preprocessing on the other 80%. On a tiny table (the last
        # steps of a row-removal sweep) that leaves a single row, every column is then
        # constant and dropped, and the model gets a table without features. Giving the
        # training set as its own validation set turns the split off: the preprocessing sees
        # every row, and the result no longer depends on a random split.
        n_varying = sum(len(set(column)) > 1 for column in np.asarray(X_train).T)
        if n_varying == 0:
            raise ValueError("Mitra drops the constant columns, and none of them varies")
        self.clf.fit(X_train, y_train, X_val=X_train, y_val=y_train)
        return self

    def default_layers(self):
        return self.stack_layers() + ["last"]

    def _named_readouts(self):
        return {"last": self._module_readout(self.FINAL_MODULE)}

    def _module_readout(self, name):
        import torch

        if name != self.FINAL_MODULE and not re.fullmatch(r"layers\.\d+", name):
            raise ValueError(
                f"Mitra can be read after a layer ('layers.<i>' or its index) or at "
                f"'{self.FINAL_MODULE}', not at '{name}'"
            )

        def reduce(capture):
            # One forward pass per chunk of query rows, in order.
            activations = torch.cat(capture.chunks, dim=1)  # (1, query rows, features + 1, dim)
            if activations.ndim != 4 or activations.shape[1] != capture.n_test:
                raise RuntimeError(
                    f"Captured {tuple(activations.shape)} at {name}, expected "
                    f"(1, {capture.n_test}, features + 1, dim)"
                )
            if self.feature_slot == "target":
                return activations[:, :, 0, :]
            if self.feature_slot == "mean":
                return activations[:, :, 1:, :].mean(dim=-2)
            if self.feature_slot == "flatten":
                return activations[:, :, 1:, :].flatten(start_dim=-2)
            raise ValueError(f"Unknown feature_slot {self.feature_slot}")

        # A layer returns the pair (support, query), the query rows are the ones to embed.
        return Readout(name, reduce, pick=1)
