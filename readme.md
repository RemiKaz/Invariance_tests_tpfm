# Tests exps SAE x Tabular PFM

Embedding-invariance experiments on tabular foundation models: a model is fitted on a
perturbed version of a dataset, its test-row embeddings are read at a chosen layer, and
they are compared to the embeddings obtained without perturbation (`--metric`, cosine
similarity by default).

## Environment

On Jean Zay the venv sits on top of the IDRIS `pytorch-gpu` module, which provides torch
built against the CUDA/NCCL of the machine. The module has to be loaded before the venv is
activated, `setup_env.sh` does both:

```bash
source setup_env.sh
```

`job.slurm` is the matching batch script (`mkdir -p logs` once, then `sbatch job.slurm`).

### LimiX-2 (`limix_v2`)

LimiX is not on PyPI. Clone it, install it without its dependencies (it asks for `torch>=2.9.1`,
and pip would replace the module's torch, as for Mitra) with the few light ones it needs, and
download the checkpoint (1.6 GB, StableAI non-commercial licence) on a login node:

```bash
git clone https://github.com/limix-ldm-ai/LimiX.git && cd LimiX
pip install --no-deps -e . && pip install nvtx kditransform einops
python down_weight.py          # from the project: also fetches LimiX-2.ckpt into $HF_HOME
```

Without installing, `LIMIX_ROOT=/path/to/LimiX` works too. It needs Python 3.12; the module's
torch is 2.8.0, below the 2.9.1 LimiX lists, which has not been tried. LimiX has a top-level
package `utils` like this project: the wrapper makes the two share one (`models/limix_model.py`)
and checks that no module is named alike. Its predictor also makes a cache folder under a hard-coded
`/mnt/public/...` path, which the wrapper redirects to a temporary folder (`$LIMIX_CACHE_DIR`).

## Usage

```bash
python run_exps.py --model tabpfn_v3 --dataset breast_cancer --exp all --metric ccos nmi
```

A run reads every layer of the model (`--layer all`, the default) from the same forward passes
(hooks on every layer, one `predict_proba` per perturbation level), and writes every layer and
metric to one file, `results/runs/<model>/<dataset>/<exp>.json` (format in `utils/results.py`).
A rerun only computes the (layer, metric) pairs the file does not hold yet, so a run killed by
a time-out carries on where it stopped; `--force` recomputes them. A layer that cannot be read is
reported and left out, the others are saved and the process exits with an error.

| Argument | Values | Default |
| --- | --- | --- |
| `--model` | `tabpfn_v3`, `tabpfn_v2`, `tabicl_v2`, `tabicl_v1`, `tabfm_v1`, `mitra_v1`, `mitra_v2`, `limix_v2` | `tabpfn_v3` |
| `--dataset` | `breast_cancer`, `iris`, `wine`, `digits`, `phoneme`, `spambase` (OpenML, downloaded by `down_weight.py`), or the name of a table stored by `dataset_generator` or `dataset_generator.causal_chambers` (`cc_lt_ref_<target>_<k>`) | `breast_cancer` |
| `--exp` | one experiment or a group (see below) | `perturbations` |
| `--layer` | `all`, or one or several of: `last`, a stage or block name, an index in the layer stack, a dotted submodule name (see `--list_layers`) | `all` |
| `--metric` | one or several of `cosine`, `ccos`, `nmi`, `dcka`, `hsic` | `cosine` |
| `--n_levels` | number of perturbation intensities swept | `20` |
| `--seed` / `--split_seed` | perturbation seed / train-test split seed | `0` / `42` |
| `--results_dir` | where `runs/` and the fused figures `_fused/` go | `results` |
| `--force` | recompute the results already saved | off |
| `--eval` | also print the ROC AUC on the clean data (the clean test accuracy is always printed and saved as `test_accuracy` in the json) | off |

A result file only holds results computed with the same `--model_path`, `--n_estimators`,
`--feature_slot`, `--test_size`, `--n_levels`, `--seed` and `--split_seed`; a run with other
values stops, use another `--results_dir` (or `--force` to replace them).

### Campaign on Jean Zay

```bash
bash scripts/dev_watch.sh                # one dev-QOS job per pending (model, dataset), until all are done
MODELS="mitra_v2" DATASETS="iris" sbatch job.slurm   # one pair, dev QOS
python fuse_plot.py --all                # the figures, in results/_fused/
```

The models, datasets and metrics of the campaign are in `scripts/config.sh`. What is left to run
is read from the result files by `python3 utils/results.py pending ...` (standard library only,
it works on a login node); a pair is done once a `--layer all` run of it holds every metric of
`METRICS_ALL` for every layer.

### Metric

`--metric` picks how the perturbed embeddings are compared to the unperturbed baseline, in
`utils/metrics.py`:

- `cosine`: cosine similarity, computed per test row then averaged. Needs nothing beyond one
  row's two vectors.
- `ccos`: the same after subtracting the baseline's mean embedding from both.
- `nmi`: mutual information, estimated one embedding dimension at a time (a k-NN estimator,
  `sklearn.feature_selection.mutual_info_regression`) over the whole test set, divided by the
  baseline's MI with itself and averaged over dimensions (past `MI_MAX_DIMS` = 64 dimensions, a
  fixed random subset). It is 1 when unchanged, whatever the dataset, and the most sensitive
  metric to small changes (1 % of noise already brings it to ~0.55).
- `dcka`: debiased Centered Kernel Alignment (Kornblith et al. 2019, unbiased HSIC estimator),
  reading every dimension jointly. It stays about 0 for independent embeddings (slightly
  negative is estimation noise) and 1 for identical ones.
- `hsic`: the same unbiased HSIC as `dcka`, divided by the baseline's own HSIC only (CKA also
  divides by the perturbed embedding's): 1 when unchanged, about 0 when independent, and it also
  drops when a perturbation weakens the structure of the perturbed embedding itself.

The raw `mi` and the biased `cka` are no longer computed: raw MI tops out near log(n_test) for
unchanged embeddings (2.6 nats on iris, 3.6 on digits), and the biased CKA reads two independent
embeddings as similar when they are wide against n_test (0.78 at 50 rows x 64 dims, 0.99 at
188 x 4096), so both scales shift from one dataset and layer to the next.

**Comparing layers of one model on one dataset** (the fused plots): use `ccos` and `nmi`. In
synthetic checks with a known answer they kept the right order between layers of different width
and shape every time. Plain `cosine` inverted it whenever a layer had a large mean vector shared
by all samples (common in transformers); `ccos` subtracts the baseline mean first and does not.
CKA (any variant) inverted it when layers differed in width: it is dominated by the top principal
directions, so its value for the same damage depends on the layer's spectrum. Along one curve
(one layer, increasing levels) every metric orders correctly.

What each one measures: `cosine` whether each sample keeps its own embedding (the most
literal "how much did it move"); `dcka` whether the geometry among samples is kept (blind to a
rotation, rescaling or sign flip of the whole space); `nmi` whether each dimension still
carries the same information (blind to rescaling or a sign flip, not to a rotation).

`--metric` takes several values: they are all computed from the same embeddings in one sweep.
`fuse_plot.py` also takes `--metric` (default: `cosine` for a single plot, every metric found
for `--all`); results of different metrics are never mixed into the same figure. It does not
draw a figure again when it is newer than its result file, the table it takes the roles from,
`fuse_plot.py` and `utils/plots.py`; `--force` draws it anyway, and a choice of `--layers` or
`--full_scale` always redraws.

### Experiments

`row_shuffle`, `col_shuffle`, `random_labels`, `col_remove`, `row_remove`, `value_noise`,
`alt_target` (predict another column of X instead of y), plus two groups: `perturbations`
(the 2x3 grid of the original `test.py`) and `all`.

#### Causal chain experiments

On a table drawn by `dataset_generator` (a causal chain, stored in
`/lustre/fswork/projects/rech/ycq/ufb58bn/DataRemi/custom_scm_data` or `$CUSTOM_SCM_DATA`), the
group `chain` perturbs the causal model itself and sweeps *where* in the chain. The x axis is
the node (or edge) perturbed; the descendants are recomputed through the stored edges and noise,
and the table is rebuilt with the same train/test split (the target keeps its class cuts).

- `chain_noise`: extra Gaussian noise on one node (1 std of that node),
- `chain_randomize`: one node permuted across the rows (distribution kept, link to its parent cut),
- `chain_edge_shuffle`: instance-wise, half of the rows get the parent values of another row on
  the edges into one node (level k is node k+1, the edge from node k in a chain).

The group `chain_local` runs the same three perturbations without recomputing the descendants
(`chain_noise_local`, `chain_randomize_local`, `chain_edge_shuffle_local`). In `chain`, a node
early in the chain changes every column after it, so its position and the number of columns it
changes go together. In `chain_local` exactly one column changes wherever it is (the child for
the edge shuffle, only `y` when it is the target), which separates the two. `chain_all` runs the noise and randomize experiments both ways (the edge
shuffles are not in it: they fail on a DAG with several roots). The group `blanket` is the two column-only ones alone
(`chain_noise_local`, `chain_randomize_local`), what runs on real measurements (Causal Chambers, below).

```bash
python -m dataset_generator.dataset_generator --n_features 8 --n_classes 3 --seeds 0
python run_exps.py --model tabicl_v2 --dataset chain_8f_3c_last_seed0 --exp chain
python run_exps.py --model mitra_v2 --dataset chain_8f_3c_last_seed0 --exp chain_local
```

#### Causal DAGs and the Markov blanket of the target

`--structure dag` draws a random causal graph instead of a chain: node k has `--in_degree` parents
among the nodes before it (a chain is in-degree 1 on the previous node), each edge with a random
sign and a weight bounded away from 0. `--target blanket` (the DAG default) picks a target that has a
parent, a child and a spouse (another parent of one of its children) that is neither, so its
Markov blanket has every role. The feature columns are shuffled (`--no_permute_columns` keeps node
order), so a column's position says nothing about its node's place in the graph; the json holds the
column-to-node map (`feature_nodes`), the blanket (`markov_blanket`), each node's role
(`node_roles`: target, parent, child, spouse, outside) and its depth and number of ancestors and
descendants (`node_info`). Chain tables stored before are read the same way, nothing changes for them.

`--plot_graph` draws the graph next to the table (`<name>_graph.png`, or
`python -m dataset_generator.plot_graph --name <name>`): nodes by depth, coloured and shaped by role,
the column of `X` under each node. The experiments are the ones above, `chain_all` being the
propagated and the column-only versions of the noise and randomize perturbations, whose contrast tells whether the last layer reacts to
the blanket or to any column: a column-only perturbation of a node outside the blanket should not
move a layer that follows the blanket, while a propagated one reaches the blanket through the graph.

```bash
python -m dataset_generator.dataset_generator --structure dag --n_features 8 --n_classes 3 --noise_std 0.3 --noise_mode relative --plot_graph
python run_exps.py --model mitra_v2 --dataset dag_8f_3c_blanket_deg2_relnoise_seed0 --exp chain_all --layer 0 5 11 last
python fuse_plot.py --model mitra_v2 --dataset dag_8f_3c_blanket_deg2_relnoise_seed0 --exp chain_all --metric ccos --by_role
```

#### Groups of tables for the P(y | X) experiment

`python -m dataset_generator.cond_groups --n_groups 3 --seed 0` stores groups of five tables
(`cg00_baseline`, `cg00_markov_equiv`, `cg00_same_cond`, `cg00_anticausal`, `cg00_near_graph`, then
`cg01_...`). They are built to ask whether a model follows P(y | X) rather than the causal graph, and
they live in a linear Gaussian model (`dataset_generator/gaussian_scm.py`) where the target is a
continuous node cut into classes at fixed thresholds, so that P(X, y), P(y | X) and the Markov
equivalence class are known exactly and each relation holds by construction:

| Table | Relation to the baseline |
| --- | --- |
| `baseline` | a random DAG whose target has a parent, a child and a spouse |
| `markov_equiv` | same P(X, y), another graph of the Markov equivalence class (edges reversed, some roles change) |
| `anticausal` | same P(X, y), the graph with the target first and the features as its descendants (y -> X) |
| `same_cond` | same P(y \| X), another P(X) and another graph (a new DAG among the features, the target a child of the baseline's blanket) |
| `near_graph` | one edge into the target added (or removed): the graph is at distance 1, P(y \| X) changes |

The tables are independent samples, the same variable sits in the same column in all of them, and
the nodes of each are numbered in its own topological order, so the perturbation experiments
(`--exp chain_all`) propagate through that table's graph. Each json holds, under `group`, the checks of
its relation to the baseline: structural Hamming distance, Markov equivalence, the largest difference
of the joint covariance and of the regression of the target on the features (about 1e-15 when they
are equal), the mean total variation distance between the two P(y | X), the KL divergence between the
two P(X), and the calibration error of the baseline's P(y | X) measured on the table's own sample.
`--plot_graph` draws the graphs, `--skip_existing` leaves stored tables alone, `--names_only` prints
the names.

The experiment on these tables is `context_swap` (group `context`), run on the baseline of a group:
the model reads the baseline's test rows (the queries) with the context, i.e. the training table it is
fitted on, taken from each of the other four tables (level 0 is the baseline's own context, the
similarity is 1). The queries are the same rows, so the embeddings are aligned row by row and every
metric applies. If the embeddings follow P(y | X, context), the tables that share the baseline's
P(y | X) (`markov_equiv`, `anticausal`, `same_cond`) should leave them alone, up to the sampling noise
of the context, and `near_graph` should not. `markov_equiv` and `anticausal` are samples of the
baseline's own distribution, so they show that sampling noise.

```bash
python -m dataset_generator.cond_groups --n_groups 5
python run_exps.py --model mitra_v2 --dataset cg00_baseline --exp context_swap --layer 0 last --metric ccos
python compare_groups.py --model mitra_v2 --metric ccos        # all the groups found, by layer
STRUCTURE=groups MODELS=mitra_v2 bash scripts/pipeline.sh generate   # or run / plot, the whole thing
```

`compare_groups.py` draws, for each layer, the mean similarity over the groups for each of the four
tables, with the range between the groups (`--csv` writes the numbers).

`fuse_plot.py --by_role` puts the layers on the x axis and draws one curve per role of the perturbed
node (mean over the nodes of that role). `bash scripts/pipeline.sh` does all of it, from the data
to the figures: `STRUCTURE=dag IN_DEGREE=2 bash scripts/pipeline.sh generate`, then `run`, then `plot`.

#### Causal Chambers: the Markov blanket on real measurements

The DAG experiments suggest that a model's last layers react to the columns of the target's Markov
blanket and not to the others. `dataset_generator/causal_chambers.py` tests it on real data with a
known graph: the light tunnel of the Causal Chambers (Gamella, Peters & Bühlmann, Nature Machine
Intelligence 2025, https://causalchamber.org), a physical device whose actuators (light source
`red`/`green`/`blue`, polarisers `pol_1`/`pol_2`, LEDs `l_11` ... `l_32`) and sensors (`current`,
`angle_1`/`angle_2`, light sensors `ir_1` ... `vis_3`) follow a published ground-truth graph. The
rows come from `uniform_reference` of `lt_interventions_standard_v1` (10 000 independent draws of
the actuators, sensor parameters fixed): 20 varying variables, the actuators being the parents of
the sensors. The wind tunnel is left out, its datasets are time series of designed sequences.

A table is built per target variable: 1000 rows (disjoint across the `--n_tables` tables of a
target), the target cut into 3 equal-frequency classes, the 19 other variables as shuffled
columns. Each table stores two truths:

- `parents`, the published graph;
- `effective_parents`, its edges with a measurable effect in these rows (partial R^2 >= 1e-3 in a
  spline regression of the child on its parents, `edge_strength`). The LEDs `l_ij` have a published
  edge into the sensors of their board but no visible effect at the ranges of the reference
  experiment (partial R^2 ~ 1e-4), so no method can find a blanket through them.

The default targets are the 14 variables with a measurable blanket (every one but the LEDs): for a
sensor the blanket is its parents, for an actuator its children and their other parents. Other
sensors lit by the same source are strongly associated with the target without being in its
blanket (`ir_1` and `ir_2`: |corr| = 0.9), which is what separates a blanket from an association.
Only the column-only perturbations run on these tables (`--exp blanket`): a physical system has
no equations to recompute the descendants of a perturbed node with, and the propagated
experiments refuse such a table.

The tables are a benchmark: `benchmark_blanket.py` evaluates every method on every table in one
run and writes a leaderboard, it draws nothing. Each detector of `detectors/` says which columns
are in the blanket:

- a model, through its embedding: how much perturbing one column alone (noise, or a permutation
  of its values) moves the final-layer embedding of the test rows (`ccos`). The model is loaded
  once and run on every table, one fit per column and perturbation, final layer only; the scores
  are cached in `results/blanket_scores/<model>/<table>.json`, so a job killed by its time limit
  carries on where it stopped. `--layers output` reads the prediction instead (how much the
  predicted class probabilities move, total variation distance): permutation importance on the
  output, the reference that tells whether the embedding knows more than the prediction.
  `--probe` adds the variants of `detectors/pfm_probe.py`, from one pass per perturbation and
  seed (cached in `results/blanket_probe/`): plain, conditional (`condrandomize`) and noise
  perturbations, 1 or 3 seeds, five readouts (`ccos`, `cosine`, `dcka`, `l2`, `decoder_q`) and
  three selection rules, two of them on 6 shadow columns (permuted copies of real columns,
  outside the blanket by construction). On the light tunnel (TabPFN v3, effective truth) the best
  is the plain permutation, 3 seeds, `dcka`, cut above the shadows' mean + 2 std: top-k 0.68 and
  F1 0.56, against 0.61 and 0.38 for the single-seed `ccos` with the largest-gap cut, and 0.67 /
  0.61 for IAMB. The conditional permutation ranks worse (AUC 0.77): a blanket column that other
  columns measure almost exactly (the light source and its sensors) is barely moved by it.
  `--probe noise` sweeps the noise perturbation (`ChainNoise`: `strength` in stds of the column,
  `kind` gaussian / uniform / laplace, `rows` all / train / test). Stronger is better (AUC with
  `dcka` 0.80 at 0.1 std, 0.84 at 4 std, close to a permutation); the distribution barely matters;
  noising the queries only (`rows=test`, the context left clean, permutation importance on the
  fitted function) gives the best AUC of all (0.856) and, with the largest-gap cut, F1 0.61 as
  IAMB, but a lower top-k (0.62); the shadow cut fails there (a shadow moves nothing). Its set is cut at the largest gap of the scores, a model having no
  threshold of its own.
- IAMB, Grow-Shrink, total conditioning and the marginal association (the reference to beat),
  each with a conditional independence test (`fisher_z`, `logistic`, `spline`) fitted to the same
  context rows, on the CPU cores in parallel over the tables.

They are scored as rankings (AUC blanket vs outside, top-k precision) and as sets (precision,
recall, F1, exact recovery, recall by role), under both truths, averaged over the tables. The
outputs go to `results/benchmark/blanket_<tag>_leaderboard.md`, `_summary.csv`, `_by_target.csv`
and `.csv` (one row per table, detector and truth). The same script runs on the simulated DAGs
(`--tables 'dagpool_*' --truths graph`), and `--source results` reads a model from the curves of
`run_exps.py --exp blanket` instead of running it; the curves, every layer and metric, are for
looking at a model, not for the benchmark.

```bash
python -m dataset_generator.causal_chambers                     # on a login node: download, 14 targets x 3 tables
source $WORK/setup_env.sh && MODELS=tabpfn_v3 sbatch scripts/benchmark.slurm   # the whole benchmark, one GPU job
# or the same through the pipeline:
STRUCTURE=chamber bash scripts/pipeline.sh generate
STRUCTURE=chamber MODELS=tabpfn_v3 SCRIPT="pipeline.sh run" sbatch job.slurm
STRUCTURE=chamber bash scripts/pipeline.sh plot                 # prints the leaderboard
```

The tables go to `/lustre/fswork/projects/rech/ycq/ufb58bn/DataRemi/causal_chamber_data` (or
`$CAUSAL_CHAMBER_DATA`), the downloaded zip (3.9 MB, MD5-checked) to its `raw/`. Neither the
`causalchamber` package nor pandas is needed: the ground-truth edges are copied from the package
(MIT licence).

### Layer position

`--layer last` uses the embeddings the model exposes itself (`get_embeddings` for TabPFN,
the end of the ICL transformer for TabICL). Any other value reads the activations with a forward
hook, either by index in the layer stack or by submodule name. The names depend on the
installed version of the library, so list them first:

```bash
python run_exps.py --model tabpfn_v3 --list_layers
python run_exps.py --model tabpfn_v3 --layer 5
python run_exps.py --model tabicl_v2 --layer row_interactor.layers.3
```

TabICL (`tabicl_v2`, `tabicl_v1`) and TabPFN v3 (`tabpfn_v3`) have three named layer
positions, in the order the data flows through the network:

| `--layer` | Reads | Shape per row |
| --- | --- | --- |
| `tf_col` | column-wise stage output (TabICL: `col_embedder.tf_col`, before the affine map to cell embeddings; TabPFN: `feature_distribution_embedder`) | `features x dim`, reduced by `--feature_slot` |
| `tf_row` | row-wise stage output, the concatenated CLS tokens (TabICL: `row_interactor`; TabPFN: `column_aggregator`) | `n_cls x dim` |
| `tf_icl` | end of the in-context-learning transformer, after its final norm, what the decoder reads (TabICL: `icl_predictor.ln`; TabPFN: `output_norm`) | `dim` |

For TabICL only, `tf_col` and `tf_row` are each a stack of blocks, and `--layer tf_col_early` /
`tf_col_middle` and `tf_row_early` / `tf_row_middle` read an earlier block instead of the last
one (closing the first quarter / half of that stack), the same way as the stage's own final
output: `tf_col_*` reduces the block's per-cell output the same way `tf_col` does, `tf_row_*`
reads the same CLS-token slots `tf_row` reads, just refined less. `tf_row`'s last block does
the CLS-only pooling that `tf_row` itself reads, so `tf_row_early` / `tf_row_middle` are
computed over its other blocks only.

TabFM v1 (`tabfm_v1`) alternates column and row stages (`col_embedder`, `row_interactor`,
`col_embedder_2`, `row_interactor_2`, then the ICL transformer), so a bare `tf_col` or `tf_row`
would be ambiguous. Its layers are numbered in forward order: `tf_col_1`, `tf_row_1`, `tf_col_2`,
`tf_row_2` (the row output the ICL stage reads) and `tf_icl`. `tf_col_*` are the set-transformer
outputs, before the linear map and norm that follow them.

LimiX-2 (`limix_v2`) is one stack of 24 blocks over the features and the rows. A row is its feature
tokens followed by 4 target tokens (256 values each), which the decoder reads: `--layer <i>` (0 to 23)
is the output of block `i` for those target tokens of the query rows, 1024 values per row, and
`--layer last` the same after block 23's final norm, what the decoder reads. `--feature_slot` has no
effect. `--n_estimators` is the number of preprocessing pipelines of its ensemble (32 in its default
config); 1 is a single forward pass, more are averaged.

Mitra (`mitra_v1`, `mitra_v2`) is a stack of 12 identical layers, each attending across the rows
and then across the features, so it has no column or row stage to name. Read it with
`--layer <i>` (0 to 11, the output of layer `i` for the query rows), `--layer early` / `middle`
(the layers closing the first quarter and the first half of the stack, layers 2 and 5) or
`--layer last` (after the final norm, right before the prediction head). Each row is a set of `features + 1` tokens, the
first one being the target token: `--feature_slot target` (the default) keeps it, `mean` averages
the feature tokens and `flatten` concatenates them. The classifier runs in-context, without
fine-tuning.

`--layer all` (what the campaign runs, `default_layers()` of each model) reads every block
rather than the early / middle depths:

| Model | Layers |
| --- | --- |
| `tabpfn_v2`, `mitra_v1`, `mitra_v2` | `0` to `11` (the 12 layers), `last` |
| `limix_v2` | `0` to `23` (the 24 blocks), `last` |
| `tabpfn_v3` | `col_0`, `col_1`, `tf_col`, `row_0`, `row_1`, `tf_row`, `0` to `23` (the ICL blocks), `tf_icl`, `last` |
| `tabicl_v1`, `tabicl_v2` | `col_0`, `col_1`, `tf_col`, `row_0`, `row_1`, `tf_row`, `icl_0`-`icl_11`, `tf_icl` |
| `tabfm_v1` | `col1_0`, `col1_1`, `tf_col_1`, `row1_0`-`row1_2`, `tf_row_1`, `col2_0`, `col2_1`, `tf_col_2`, `row2_0`-`row2_2`, `tf_row_2`, `0` to `23` (the ICL blocks), `tf_icl` |

In a column stage the last block is the stage's own output (`tf_col`, `tf_col_1`...), so it is not listed twice. The
first row stage of TabICL and `tabpfn_v3` end with a CLS-only readout that is `tf_row` itself. `col_<i>`, `row_<i>`
and `icl_<i>` (TabICL) or `col_<i>`, `row_<i>` (`tabpfn_v3`) or `col<s>_<i>`, `row<s>_<i>` (TabFM stage `s`) read
block `i` of a stage, the CLS tokens for the row stages.

For TabICL and TabFM, `tf_icl` is also what `--layer last` reads. For `tabpfn_v3`, `last` still means
the `get_embeddings` output of the library. `tabpfn_v2` has one flat stack of alternating
attention blocks and no such stages: read it with `--layer <index>` on its `blocks`, or by
depth with `--layer early` (first quarter of the blocks), `middle` (first half) and `last`.

For `tf_col` there is no target cell (TabICL and TabPFN v3), so `--feature_slot` is `mean` (also what `target` is
read as) or `flatten`. For TabICL the 4 columns reserved for the CLS tokens are dropped. Do not hook
`row_interactor.tf_row` itself: its last block is called directly, so the hook never fires.

## Structure

```
setup_env.sh                       # module + venv, to be sourced
job.slurm                          # batch script for the compute nodes, dev QOS
run_exps.py                        # argparse entry point, saves results/runs/<model>/<dataset>/<exp>.json
fuse_plot.py                       # the fused figures: one curve per layer (or per role)
models/model_importer.py           # model_name (with version) + params -> model
models/base_model.py               # fit / predict / get_embeddings_of(layers): a Readout per layer, one pass
models/tabpfn_model.py             # shared TabPFN wrapper + HF checkpoint lookup
models/tabpfn_v3_model.py          # TabPFNV3: default checkpoint
models/tabpfn_v2_model.py          # TabPFNV2: default checkpoint
models/tabicl_model.py             # shared TabICL wrapper
models/tabicl_v2_model.py          # TabICLV2: pinned checkpoint version
models/tabicl_v1_model.py          # TabICLV1: pinned checkpoint version
models/tabfm_v1_model.py           # TabFMV1: Google TabFM, PyTorch backend
models/mitra_model.py              # shared Mitra wrapper (AutoGluon), layer hooks
models/mitra_v1_model.py           # MitraV1: autogluon/mitra-classifier
models/mitra_v2_model.py           # MitraV2: autogluon/mitra-classifier-2
models/limix_model.py              # shared LimiX wrapper: predictor import, the y tokens of the 24 blocks
models/limix_v2_model.py           # LimiXV2: stable-ai/LimiX-2 checkpoint
datasets/dataset_importer.py       # dataset_name -> TabularDataset
datasets/tabular_dataset.py        # train/test split container
datasets/custom_scm_dataset.py     # stored dataset_generator tables + their metadata
datasets/causal_chamber_dataset.py # stored Causal Chamber tables: real measurements, published + effective graph
dataset_generator/dataset_generator.py # causal chain / DAG prior -> stored table + metadata
dataset_generator/scm_graph.py     # the graph: node values, resimulation, Markov blanket, roles
dataset_generator/plot_graph.py    # draw the graph of a stored table
dataset_generator/gaussian_scm.py  # linear Gaussian models: graphs, covariances, equivalence classes
dataset_generator/cond_groups.py   # groups of five tables with a prescribed relation to a baseline
dataset_generator/causal_chambers.py # light tunnel of the Causal Chambers -> tables with a known graph (download)
experiments/experiment_importer.py # exp_name -> list of experiments
experiments/base_experiment.py     # Experiment interface + run_experiment loop
experiments/perturbations.py       # the six perturbation sweeps
experiments/target_column.py       # alternative-target sweep
experiments/causal_chain.py        # perturbations at each node of a causal chain or DAG (+ conditional permutation)
experiments/context_swap.py        # same query rows, context from another table of a group
compare_groups.py                  # context_swap over the groups: similarity by layer and table
benchmark_blanket.py               # Markov blanket benchmark: models vs IAMB, Grow-Shrink, ... on every table at once, leaderboard
detectors/detector_importer.py     # detector_name -> Markov blanket detector
detectors/base_detector.py         # scores (a ranking) + select (a set), largest-gap cut; constraint-based base
detectors/ci_tests.py              # x_j _||_ y | X_S: fisher_z, logistic, spline likelihood-ratio tests
detectors/iamb.py                  # IAMB (Tsamardinos et al. 2003)
detectors/grow_shrink.py           # Grow-Shrink (Margaritis & Thrun 1999)
detectors/total_conditioning.py    # one test per column given all the others (Pellet & Elisseeff 2008)
detectors/marginal.py              # association with the target alone: the reference to beat
detectors/pfm_perturbation.py      # a model's embedding: sensitivity to each column perturbed alone (runs the model, cached)
detectors/pfm_probe.py             # its variants: perturbations, seeds, readouts, shadow-column thresholds (one cached pass each)
utils/metrics.py                   # cosine / ccos / nmi / dcka / hsic, the --metric choices
utils/results.py                   # the result files: paths, merge, layer order, pending/accuracy CLI
utils/plots.py                     # the fused figures and their layout
utils/style.py                     # colours and markers shared by every figure
utils/utils.py                     # permutations
scripts/config.sh                  # campaign defaults: models, datasets, metrics, pending_pairs
scripts/all_runs.sh                # every pending (model, dataset) in sequence, what job.slurm runs
scripts/dev_watch.sh               # keep the dev QOS full until everything is done
scripts/pipeline.sh                # the causal study: generate, run, plot
scripts/benchmark.slurm            # the Markov blanket benchmark in one GPU job (dev QOS)
```

Adding a model version, a dataset or an experiment means adding one file and one entry in the
matching importer.

## Notes

- The TabPFN v3 (`tabpfn_v3`) and TabPFN v2 (`tabpfn_v2`) checkpoints are read from `$WORK/hf_cache`, where `down_weight.py` puts them;
  the compute nodes have no internet access, so run it on a login node first. `TABPFN_V3_CKPT` /
  `TABPFN_V2_CKPT` override the resolved path, `--model_path` overrides it per run.
- `n_estimators=1` keeps a single deterministic forward pass, so the captured
  representations are not an average over shuffled ensembles.
- The original standalone scripts this repo was factored out of (`test.py`...) are in the
  backup `../SAE_TabularPFM_test_v01.zip`, with the results in the old layout (one folder per
  layer and metric); `scripts/migrate_results.py` converts those to the current one.
- Never let pip change the torch of the module (`pip install` of a package that pins a newer
  torch, such as `autogluon.tabular[mitra]`, installs its own torch and CUDA libraries in the
  venv and breaks every model). Install such packages with `--no-deps`, see requirements.txt.
  If it happens, `pip freeze --local | grep -iE '^(torch|triton|nvidia|cuda)'` shows the
  intruders, and uninstalling them brings back the module's version.
