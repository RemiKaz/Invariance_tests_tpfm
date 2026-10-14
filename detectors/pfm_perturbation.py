import json
import os

import numpy as np

from detectors.base_detector import BlanketDetector
from utils.results import layer_key, load, run_path

# the experiment groups a run_exps.py run of the column-only perturbations can be saved under, in
# the order they are looked for (source="results")
RUN_EXPERIMENTS = ("blanket", "chain_all", "chain_local")

# the `layer` that reads the prediction (the class probabilities) instead of an embedding
OUTPUT = "output"

# model name -> the loaded model, shared by the detectors of one model (one per perturbation), so
# that a model is loaded once for a whole benchmark; `release_models` frees them
_LOADED = {}


def release_models():
    """Drop the loaded models and free the GPU memory they held."""
    _LOADED.clear()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


class PFMPerturbation(BlanketDetector):
    """A tabular foundation model's embedding as a Markov blanket detector: the score of a column
    is how much perturbing that column alone moves the embedding of the test rows,
    1 - similarity to the embedding of the unperturbed table. With `layer="output"` it is how
    much it moves the prediction instead: the mean total variation distance between the class
    probabilities of the test rows, 1/2 sum_c |p_c - p0_c| (0 to 1), permutation importance on
    the output.

    With `source="run"` (the default) the detector runs the model itself: one fit on the clean
    table, then one fit per feature column with that column perturbed (the perturbations of the
    `_local` experiments, see experiments/causal_chain.py), reading one layer with one metric.
    The scores are cached in `<results_dir>/blanket_scores/<model>/<table>.json`, so a benchmark
    killed by a time limit carries on where it stopped. With `source="results"` it reads instead
    the curves saved by `run_exps.py --exp blanket` (or chain_all), whose level is the perturbed
    node.

    The set is cut at the largest gap of the scores (`largest_gap`): a model has no threshold of
    its own.

    Args:
        model (str): The model, as in run_exps.py --model.
        exp (str, optional): The perturbation, chain_randomize_local (the column permuted across
            the rows) or chain_noise_local (Gaussian noise of one std). Defaults to
            "chain_randomize_local".
        metric (str, optional): The similarity, see utils/metrics.py. Defaults to "ccos".
        layer (str, optional): The layer read, "final" for the end of the model (the last of its
            default layers), or "output" for the predicted class probabilities. Defaults to
            "final".
        source (str, optional): "run" or "results", see above. Defaults to "run".
        results_dir (str, optional): Where the score cache (and run_exps.py's runs) live.
            Defaults to "results".
        device (str, optional): The device of the model. Defaults to "cuda".
        seed (int, optional): The seed of the model and of the perturbations. Defaults to 0.
    """

    def __init__(self, model, exp="chain_randomize_local", metric="ccos", layer="final", source="run",
                 results_dir="results", device="cuda", seed=0):
        if source not in ("run", "results"):
            raise ValueError(f"source must be run or results, not {source}")
        self.model, self.exp, self.metric, self.layer, self.source = model, exp, metric, layer, source
        self.results_dir, self.device, self.seed = results_dir, device, seed
        short = exp.replace("chain_", "").replace("_local", "")
        self.name = f"pfm:{model}:{short}" + ("" if layer == "final" else f"@{layer}")

    # ---- source="run": the model is run here

    def loaded(self):
        """The model, loaded once and shared with the other detectors of the same model."""
        if self.model not in _LOADED:
            from models import model_importer  # torch is only imported when a model is run

            _LOADED[self.model] = model_importer(self.model, device=self.device, seed=self.seed)
        return _LOADED[self.model]

    def cache_path(self, dataset):
        return os.path.join(self.results_dir, "blanket_scores", self.model, f"{dataset.name}.json")

    def computed(self, dataset):
        """The sensitivity of every feature column, from the cache or by running the model."""
        path = self.cache_path(dataset)
        cache = load(path) or {}
        metric = "tv" if self.layer == OUTPUT else self.metric
        key = f"{self.exp}|{metric}|{self.layer}|seed{self.seed}"
        if key in cache.get("scores", {}) and cache.get("feature_nodes") == list(dataset.feature_nodes):
            return np.array(cache["scores"][key])

        from experiments import experiment_importer
        from utils.metrics import METRICS

        model = self.loaded()
        model.fit(dataset.X_train, dataset.y_train)  # first: some models only build the network in fit
        if self.layer == OUTPUT:
            # the prediction itself: the class probabilities of the test rows, compared by their
            # mean total variation distance (--metric is for embeddings, it does not apply)
            layer = OUTPUT
            baseline = model.predict_proba(dataset.X_test)

            def change(perturbed):
                return 0.5 * float(np.abs(model.predict_proba(perturbed.X_test) - baseline).sum(axis=1).mean())
        else:
            layer = model.default_layers()[-1] if self.layer == "final" else self.layer
            clean, errors = model.get_embeddings_of(dataset.X_test, [layer])
            if errors:
                raise RuntimeError(f"{self.model} cannot read layer {layer}") from errors[layer]

            def change(perturbed):
                embeds, _ = model.get_embeddings_of(perturbed.X_test, [layer])
                return 1.0 - float(METRICS[self.metric].fn(embeds[layer], clean[layer]))
        perturbation = experiment_importer(self.exp)[0]
        rng = np.random.default_rng(self.seed)
        scores = []
        for node in dataset.feature_nodes:  # column order; the target node is not needed
            perturbed = perturbation.perturb(dataset, node, rng)
            model.fit(perturbed.X_train, perturbed.y_train)
            scores.append(change(perturbed))
        print(f"  {self.name} on {dataset.name}: {len(scores)} columns, layer {layer}")

        cache = load(path) or {}  # another detector of the model may have written meanwhile
        cache.update({"model": self.model, "table": dataset.name, "feature_nodes": list(dataset.feature_nodes)})
        cache.setdefault("scores", {})[key] = scores
        cache.setdefault("layers", {})[key] = layer
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as file:
            json.dump(cache, file, indent=1)
        return np.array(scores)

    # ---- source="results": the curves saved by run_exps.py

    def payload(self, dataset):
        """The saved run of the column-only perturbations on this table."""
        for run_exp in RUN_EXPERIMENTS:
            payload = load(run_path(self.results_dir, self.model, dataset.name, run_exp))
            if payload and any(r["name"] == self.exp for ms in payload["layers"].values()
                               for r in ms.get(self.metric, [])):
                return payload
        raise FileNotFoundError(f"No {self.exp} / {self.metric} run of {self.model} on {dataset.name} in {self.results_dir}")

    def from_results(self, dataset):
        if self.layer == OUTPUT:
            raise KeyError("run_exps.py saves embeddings only: the output needs source='run'")
        payload = self.payload(dataset)
        layers = sorted((l for l, ms in payload["layers"].items() if self.metric in ms), key=layer_key)
        layer = self.layer
        if layer == "final":
            layer = "last" if "last" in layers else "tf_icl" if "tf_icl" in layers else layers[-1]
        for result in payload["layers"][layer][self.metric]:
            if result["name"] == self.exp:
                similarity = np.array([np.nan if s is None else s for s in result["similarities"]], dtype=float)
                return np.nan_to_num(1 - similarity[dataset.feature_nodes], nan=0.0)
        raise KeyError(f"No {self.exp} at layer {layer} of {self.model} on {dataset.name}")

    def scores(self, dataset):
        return self.computed(dataset) if self.source == "run" else self.from_results(dataset)
