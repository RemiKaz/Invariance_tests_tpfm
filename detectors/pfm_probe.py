"""A model's embedding as a Markov blanket detector, with the alternatives meant to improve its
ranking (top-k) and its selected set (F1).

The scores come from one engine, `probe_scores`: for a table, a perturbation and a seed, the
model is fitted on the clean table, then once per feature column with that column perturbed
alone, and the final-layer embedding of the test rows is compared to the clean one under every
READOUT at once (one forward pass gives them all). The table is first augmented with `n_shadows`
shadow columns, copies of real columns permuted across the rows (Boruta, Kursa & Rudnicki 2010):
independent of everything, they are outside the blanket by construction, and their scores show
what a column without effect scores. Everything is cached in
`<results_dir>/blanket_probe/<model>/<table>.json`, so the variants of `PFMProbe` (which
perturbations, how many seeds, which readout, which threshold) are evaluated without running the
model again.

The alternatives:

- perturbation: `randomize` (the column permuted across the rows), `noise` (Gaussian noise of one
  std), `condrandomize` (permuted given the other columns, experiments/causal_chain.py): only a
  blanket column changes P(y | X) then, where a plain permutation also makes impossible rows.
- seeds: the scores of several draws of the perturbation averaged.
- readout: how the perturbed embedding is compared to the clean one, `ccos`, `cosine`, `dcka`
  (1 - similarity), `l2` (mean distance between a row's two embeddings over the mean distance of
  the clean embeddings to their mean), `decoder_q` (`ccos` after the query projection of the
  model's own decoder, the directions the prediction reads; TabPFN v3), `hsic` (1 - HSIC between
  the two embeddings over the clean embedding's own HSIC, utils/metrics.py: CKA without the
  normalisation by the perturbed embedding).
- several perturbations: their within-table ranks averaged.
- rule (the selected set): `gap` (the largest gap of the scores), `shadow_max` (above every
  shadow), `shadow_2sd` (above the mean of the shadows plus two of their standard deviations).
"""

import dataclasses
import json
import os
import zlib

import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split

from detectors.base_detector import BlanketDetector, largest_gap
from detectors.pfm_perturbation import _LOADED
from utils.results import load

PERTURBATIONS = {"randomize": "chain_randomize_local", "noise": "chain_noise_local",
                 "condrandomize": "chain_condrandomize_local"}
# the noise sweep (`--probe noise`): strength (in stds of the column), distribution, rows noised
NOISE_STRENGTHS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
NOISE_KINDS = ("uniform", "laplace")
NOISE_ROWS = ("train", "test")


def noise_spec(strength=1.0, kind="gaussian", rows="all"):
    """The name of a noise perturbation: "noise" for the default (1 std, Gaussian, every row,
    the runs cached before the sweep), else e.g. noise:s0.5, noise:s1:laplace, noise:s1:test."""
    if (strength, kind, rows) == (1.0, "gaussian", "all"):
        return "noise"
    parts = [f"s{strength:g}"] + ([kind] if kind != "gaussian" else []) + ([rows] if rows != "all" else [])
    return "noise:" + ":".join(parts)


def make_perturbation(spec):
    """The experiment of a perturbation name: a key of PERTURBATIONS, or a noise_spec."""
    from experiments import experiment_importer

    if spec in PERTURBATIONS:
        return experiment_importer(PERTURBATIONS[spec])[0]
    head, *parts = spec.split(":")
    if head != "noise":
        raise ValueError(f"Perturbation {spec} not implemented ! Available: {list(PERTURBATIONS)} or noise:s<std>[:kind][:rows]")
    params = {"strength": float(parts[0][1:])}
    for part in parts[1:]:
        params["rows" if part in ("all", "train", "test") else "kind"] = part
    return experiment_importer("chain_noise_local", params)[0]
READOUTS = ("ccos", "cosine", "dcka", "l2", "decoder_q", "hsic")
# the readouts of the default and noise grids: their cached passes predate hsic, so they keep the
# five they were run with (asking for hsic re-runs a pass, see probe_scores)
_GRID_READOUTS = ("ccos", "cosine", "dcka", "l2", "decoder_q")
RULES = ("gap", "shadow_max", "shadow_2sd")
# (model, table, run key) -> the scores already read or computed in this process: the variants
# of the grid read the same runs again and again
_MEMO = {}


def with_shadows(dataset, n_shadows):
    """The table with `n_shadows` shadow columns: copies of real feature nodes permuted across
    the rows, put at random positions among the columns. Deterministic for a table.

    Returns:
        tuple: the augmented dataset, the column of every original feature column in it, and
            the columns of the shadows.
    """
    rng = np.random.default_rng(zlib.crc32(dataset.name.encode()))
    n_rows, n_nodes = dataset.nodes.shape
    sources = rng.choice(dataset.feature_nodes, n_shadows, replace=n_shadows > len(dataset.feature_nodes))
    shadows = np.column_stack([dataset.nodes[rng.permutation(n_rows), s] for s in sources]) if n_shadows else \
        np.empty((n_rows, 0))
    nodes = np.hstack([dataset.nodes, shadows])
    shadow_nodes = list(range(n_nodes, n_nodes + n_shadows))
    order = list(dataset.feature_nodes) + shadow_nodes
    feature_nodes = [order[i] for i in rng.permutation(len(order))] if n_shadows else order
    X = nodes[:, feature_nodes]
    # the split of from_xy depends on the number of rows only, so it is replayed
    train, test = train_test_split(np.arange(n_rows), test_size=dataset.test_size, random_state=dataset.seed)
    augmented = dataclasses.replace(dataset, X=X, X_train=X[train], X_test=X[test], nodes=nodes,
                                    metadata={**dataset.metadata, "feature_nodes": feature_nodes})
    real = [feature_nodes.index(k) for k in dataset.feature_nodes]
    return augmented, real, [feature_nodes.index(k) for k in shadow_nodes]


def _readouts(embeds, clean, q_weight):
    """Every READOUT of a perturbed embedding against the clean one, as a change (0 = none)."""
    from utils.metrics import METRICS

    out = {name: 1.0 - float(METRICS[name].fn(embeds, clean)) for name in ("ccos", "cosine", "dcka")}
    spread = np.linalg.norm(clean - clean.mean(0), axis=1).mean()
    out["l2"] = float(np.linalg.norm(embeds - clean, axis=1).mean() / spread)
    out["hsic"] = 1.0 - float(METRICS["hsic"].fn(embeds, clean))
    if q_weight is not None:
        w, b = q_weight
        out["decoder_q"] = 1.0 - float(METRICS["ccos"].fn(embeds @ w.T + b, clean @ w.T + b))
    return out


def _decoder_query(model):
    """The weight and bias of the query projection of the model's decoder, or None (TabPFN v3's
    many-class decoder reads the final embedding through it)."""
    torch_model = model._torch_model()
    decoder = getattr(torch_model, "many_class_decoder", None)
    if decoder is None or not hasattr(decoder, "q_projection"):
        return None
    layer = decoder.q_projection
    return layer.weight.detach().float().cpu().numpy(), layer.bias.detach().float().cpu().numpy()


def probe_scores(model_name, dataset, perturbation, seed, n_shadows=6, results_dir="results", device="cuda",
                 need=None):
    """The change of the final-layer embedding when each column of the shadow-augmented table is
    perturbed alone, under every READOUT, cached.

    Args:
        model_name (str): The model, as in run_exps.py --model.
        dataset (CustomSCMDataset): The table (simulated or Causal Chamber).
        perturbation (str): A key of PERTURBATIONS.
        seed (int): The seed of the perturbation (and of the model).
        n_shadows (int, optional): Shadow columns added to the table. Defaults to 6.
        results_dir (str, optional): Where the cache goes. Defaults to "results".
        device (str, optional): The device of the model. Defaults to "cuda".
        need (str, optional): A readout the caller needs: a cached pass without it (run before
            that readout existed) is run again. Defaults to None, any cached pass will do.

    Returns:
        tuple: readout -> scores over the columns of the augmented table, the columns of the
            original feature columns in it, and the columns of the shadows.
    """
    augmented, real, shadow = with_shadows(dataset, n_shadows)
    path = os.path.join(results_dir, "blanket_probe", model_name, f"{dataset.name}.json")
    key = f"{perturbation}|seed{seed}|shadows{n_shadows}"
    memo = (model_name, dataset.name, key)
    if memo in _MEMO and (need is None or need in _MEMO[memo]):
        return _MEMO[memo], real, shadow
    cache = load(path) or {}
    runs = cache.get("runs", {})
    if key in runs and runs[key]["feature_nodes"] == augmented.feature_nodes and (need is None or need in runs[key]["scores"]):
        _MEMO[memo] = {r: np.array(v) for r, v in runs[key]["scores"].items()}
        return _MEMO[memo], real, shadow

    from experiments import experiment_importer
    from models import model_importer

    if model_name not in _LOADED:
        _LOADED[model_name] = model_importer(model_name, device=device, seed=0)
    model = _LOADED[model_name]
    model.fit(augmented.X_train, augmented.y_train)  # first: some models only build the network in fit
    layer = model.default_layers()[-1]
    clean = model.get_embeddings(augmented.X_test, layer)
    q_weight = _decoder_query(model)
    experiment = make_perturbation(perturbation)
    rng = np.random.default_rng(seed)
    scores = {}
    for node in augmented.feature_nodes:
        perturbed = experiment.perturb(augmented, node, rng)
        model.fit(perturbed.X_train, perturbed.y_train)
        for name, value in _readouts(model.get_embeddings(perturbed.X_test, layer), clean, q_weight).items():
            scores.setdefault(name, []).append(value)
    print(f"  probe {model_name} {perturbation} seed {seed} on {dataset.name}: {len(augmented.feature_nodes)} columns "
          f"({n_shadows} shadows), layer {layer}")

    cache = load(path) or {}
    cache.setdefault("runs", {})[key] = {"feature_nodes": augmented.feature_nodes, "layer": layer, "scores": scores}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as file:
        json.dump(cache, file)
    _MEMO[memo] = {r: np.array(v) for r, v in scores.items()}
    return _MEMO[memo], real, shadow


class PFMProbe(BlanketDetector):
    """A model's embedding as a blanket detector, with a choice of perturbations, seeds, readout
    and selection rule (see the module docstring).

    Args:
        model (str): The model.
        perturbations (tuple, optional): Keys of PERTURBATIONS; several are combined by their
            averaged ranks. Defaults to ("randomize",).
        seeds (tuple, optional): The seeds averaged. Defaults to (0,).
        readout (str, optional): A key of READOUTS. Defaults to "ccos".
        rule (str, optional): A key of RULES. Defaults to "gap".
        n_shadows (int, optional): Shadow columns. Defaults to 6.
        results_dir (str, optional): Where the cache is. Defaults to "results".
        device (str, optional): The device of the model. Defaults to "cuda".
    """

    def __init__(self, model, perturbations=("randomize",), seeds=(0,), readout="ccos", rule="gap", n_shadows=6,
                 results_dir="results", device="cuda"):
        if readout not in READOUTS or rule not in RULES:
            raise ValueError(f"readout in {READOUTS}, rule in {RULES}")
        self.model, self.perturbations, self.seeds = model, tuple(perturbations), tuple(seeds)
        self.readout, self.rule, self.n_shadows = readout, rule, n_shadows
        self.results_dir, self.device = results_dir, device
        self.name = f"probe:{model}:{'+'.join(self.perturbations)}x{len(self.seeds)}:{readout}:{rule}"

    def combined(self, dataset):
        """The score of every column of the augmented table, and the real and shadow columns."""
        per_perturbation = []
        for perturbation in self.perturbations:
            runs = []
            for seed in self.seeds:
                scores, real, shadow = probe_scores(self.model, dataset, perturbation, seed, self.n_shadows,
                                                    self.results_dir, self.device, need=self.readout)
                if self.readout not in scores:
                    raise KeyError(f"{self.model} has no readout {self.readout}")
                runs.append(scores[self.readout])
            per_perturbation.append(np.mean(runs, axis=0))
        if len(per_perturbation) == 1:
            return per_perturbation[0], real, shadow
        # the scales of two perturbations differ: their ranks are averaged instead
        return np.mean([rankdata(s) / len(s) for s in per_perturbation], axis=0), real, shadow

    def scores(self, dataset):
        combined, real, _ = self.combined(dataset)
        return combined[real]

    def select(self, dataset):
        combined, real, shadow = self.combined(dataset)
        scores = combined[real]
        if self.rule == "gap" or not shadow:
            return largest_gap(scores)
        null = combined[shadow]
        threshold = null.max() if self.rule == "shadow_max" else null.mean() + 2 * null.std()
        return scores > threshold


def probe_grid(model, n_shadows=6, results_dir="results", device="cuda", grid_name="default"):
    """The variants the benchmark evaluates (`--probe <grid_name>`).

    "default": every readout and rule for the plain and the conditional permutation (1 and 3
    seeds), the noise, and two combinations. "noise": every readout and rule for the noise at
    each of NOISE_STRENGTHS (Gaussian, every row), and at 1 std for each of NOISE_KINDS and of
    NOISE_ROWS, one seed each. "hsic": hsic against dcka, every rule, on the configurations the two
    first grids found best (permutation with 1 and 3 draws, noise of 4 s.d., noise on the queries
    only, noise of 1 s.d.): 6 passes, re-run since their cache predates hsic.
    """
    if grid_name == "noise":
        specs = [noise_spec(s) for s in NOISE_STRENGTHS] + [noise_spec(kind=k) for k in NOISE_KINDS] \
            + [noise_spec(rows=r) for r in NOISE_ROWS]
        return [PFMProbe(model, (spec,), (0,), readout, rule, n_shadows, results_dir, device)
                for spec in specs for readout in _GRID_READOUTS for rule in RULES]
    if grid_name == "hsic":
        # HSIC against dCKA on the configurations the search kept (Section 5.2 of the proposal)
        configs = [(("randomize",), (0,)), (("randomize",), (0, 1, 2)), ((noise_spec(4.0),), (0,)),
                   ((noise_spec(rows="test"),), (0,)), (("noise",), (0,))]
        return [PFMProbe(model, perts, seeds, readout, rule, n_shadows, results_dir, device)
                for perts, seeds in configs for readout in ("hsic", "dcka") for rule in RULES]
    if grid_name != "default":
        raise ValueError(f"Probe grid {grid_name} not implemented ! Available: ['default', 'noise', 'hsic']")
    grid = []
    for perturbations, seed_options in ((("randomize",), ((0,), (0, 1, 2))), (("condrandomize",), ((0,), (0, 1, 2))),
                                        (("noise",), ((0,),)), (("randomize", "condrandomize"), ((0, 1, 2),)),
                                        (("randomize", "condrandomize", "noise"), ((0,),))):
        for seeds in seed_options:
            for readout in _GRID_READOUTS:
                for rule in RULES:
                    grid.append(PFMProbe(model, perturbations, seeds, readout, rule, n_shadows, results_dir, device))
    return grid
