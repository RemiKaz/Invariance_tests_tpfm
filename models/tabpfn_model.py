import glob
import os

import numpy as np

from models.base_model import NATIVE, BaseTabularPFM


def find_hf_checkpoint(repo_id, filename, env_var):
    """Locate a checkpoint in the HF cache, whatever the snapshot hash is.

    The checkpoints are downloaded by down_weight.py, the compute nodes have no internet
    access. The cache is resolved from $HF_HOME (else $WORK/hf_cache) so it follows the
    default project, and `env_var` overrides the whole path.

    Args:
        repo_id (str): The HuggingFace repo of the checkpoint.
        filename (str): The name of the checkpoint file in the repo.
        env_var (str): The environment variable that overrides the path.

    Returns:
        str: The path of the checkpoint, or the path it is expected at when not found.
    """
    if os.environ.get(env_var):
        return os.environ[env_var]
    snapshots = os.path.join(
        os.environ.get("HF_HOME") or os.path.join(os.environ.get("WORK", ""), "hf_cache"),
        "hub",
        "models--" + repo_id.replace("/", "--"),
        "snapshots",
    )
    found = sorted(glob.glob(os.path.join(snapshots, "*", filename)))
    return found[-1] if found else os.path.join(snapshots, "<hash>", filename)


class TabPFN(BaseTabularPFM):
    """Shared TabPFN wrapper, whose final embeddings are exposed by `get_embeddings`.

    `last` is the output of the library's own `get_embeddings`, a call of its own; every other
    layer is read with a hook. The versions live in tabpfn_v2_model.py and tabpfn_v3_model.py.
    """

    # Tried in order, the first one found gives the layer stack of the transformer.
    LAYER_CONTAINERS = (
        "transformer_encoder.layers",
        "transformer_encoder",
        "encoder.layers",
        "model.transformer_encoder.layers",
    )

    def _build_classifier(self):
        from tabpfn import TabPFNClassifier

        return TabPFNClassifier(
            model_path=self.model_path,
            device=self.device,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            **self.dict_params,
        )

    def default_layers(self):
        return self.stack_layers() + ["last"]

    def _named_readouts(self):
        return {"last": NATIVE}

    def _native_embeddings(self, X_test):
        embeddings = np.asarray(self.clf.get_embeddings(X_test))
        if embeddings.ndim == 3:
            # TabPFN returns (n_estimators, n_test, dim): average the ensemble members, as the
            # hooked layers do, so every layer hands the metrics one (n_test, dim) matrix.
            embeddings = embeddings.mean(axis=0)
        return embeddings
