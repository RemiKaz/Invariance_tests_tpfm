#!/bin/bash
# The whole causal study, from the data to the figures, in three stages:
#
#   1. generate   draw the causal chain (or DAG) tables and store them (numpy only, any node)
#   2. run        the chain experiments on every layer of every model  (GPU: needs the env)
#   3. plot       one figure per model, table and metric, all layers   (matplotlib only)
#
# The clean test accuracy of every model on every table is printed at the end of stage 2 (and
# by `bash scripts/pipeline.sh accuracy` alone).
#
# Usage:
#     bash scripts/pipeline.sh generate          # on a login node
#     SCRIPT="pipeline.sh run" sbatch job.slurm  # on a GPU: the job sets up the env, then runs stage 2
#     bash scripts/pipeline.sh plot              # on a login node, once the job is done
#     bash scripts/pipeline.sh all               # the three in a row, inside an allocation
#
# Every stage skips what is already done, so a stage can be rerun after a time-out.
# The settings below can be overridden from the command line, for example:
#     SEEDS="0 1" MODELS="tabicl_v2" bash scripts/pipeline.sh generate
#     EXP=chain_local SCRIPT="pipeline.sh run" sbatch job.slurm   # the non-propagated variants
#
# STRUCTURE=dag runs it on random causal DAGs instead of chains: every node has IN_DEGREE
# parents, the target is a node with a parent, a child and a spouse (so its Markov blanket has
# every role), the columns are shuffled, and both the propagated and the column-only
# perturbations are run (EXP=chain_all). The generate stage also draws the graph of each table
# (next to it in the data folder), and the plot stage adds a figure by role of the perturbed
# node (--by_role in fuse_plot.py: parent, child, spouse, outside the blanket).
#     STRUCTURE=dag bash scripts/pipeline.sh generate
# STRUCTURE=pool does the same on POOL_SIZE (20) diverse DAG tables, each with its own number of
# features, in-degree, roots, activation, noise and classes (no graph figure per table).
#     STRUCTURE=pool MODELS=mitra_v2 bash scripts/pipeline.sh plot
# STRUCTURE=groups is the P(y | X) experiment: N_GROUPS (5) groups of five tables (a baseline, and
# tables with the same joint, the same P(y | X), the opposite direction, and a near graph, see
# dataset_generator/cond_groups.py). The baseline of each group is run with EXP=context_swap, the
# same query rows read with the context of each of the other tables, and the plot stage draws
# compare_groups.py: similarity to the baseline's own embeddings, by layer, over the groups.
#     STRUCTURE=groups MODELS=mitra_v2 bash scripts/pipeline.sh generate
# STRUCTURE=chamber is the Markov blanket benchmark on real measurements: the light tunnel of the
# Causal Chambers (dataset_generator/causal_chambers.py, downloaded at the generate stage, so on a
# login node), N_TABLES tables of N_SAMPLES rows per target variable (TARGETS, default: every one
# with a measurable Markov blanket). Its run stage is one benchmark over all the tables at once
# (benchmark_blanket.py: the models of MODELS against IAMB, Grow-Shrink, ...), which needs a GPU;
# the plot stage prints its leaderboard.
#     STRUCTURE=chamber bash scripts/pipeline.sh generate
#     STRUCTURE=chamber MODELS=tabpfn_v3 SCRIPT="pipeline.sh run" sbatch job.slurm
#     STRUCTURE=chamber bash scripts/pipeline.sh plot

# ---- Settings ----------------------------------------------------------------------------

SEEDS=${SEEDS:-"0 1"}              # one table per seed
N_SAMPLES=${N_SAMPLES:-1000}         # rows of a table
N_FEATURES=${N_FEATURES:-8}          # feature columns; the chain has one node more, the target
N_CLASSES=${N_CLASSES:-3}
STRUCTURE=${STRUCTURE:-chain}        # chain, dag, pool (POOL_SIZE diverse DAG tables, see
POOL_SIZE=${POOL_SIZE:-20}           # dataset_generator/prior_pool.py: dagpool_00, dagpool_01, ...)
N_GROUPS=${N_GROUPS:-5}              # groups: number of groups of five tables, cg00_baseline, ...
IN_DEGREE=${IN_DEGREE:-2}            # dag: parents per node
NOISE_MODE=${NOISE_MODE:-relative}   # every node has the same noise power (absolute: same std)
N_TABLES=${N_TABLES:-3}              # chamber: tables per target, on disjoint rows
TARGETS=${TARGETS:-auto}             # chamber: target variables (auto: every measurable blanket)
CHAMBER_EXPERIMENT=${CHAMBER_EXPERIMENT:-uniform_reference}   # chamber: the rows' experiment
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-4}}   # chamber: tables run in parallel by the classical detectors
if [ "$STRUCTURE" = chamber ]; then
    EXP=${EXP:-blanket}              # run_exps.py only (the curves): the column-only perturbations
elif [ "$STRUCTURE" = groups ]; then
    EXP=${EXP:-context_swap}         # the generator has its own settings, see cond_groups.py
elif [ "$STRUCTURE" != chain ]; then
    TARGET=${TARGET:-blanket}        # a node with a parent, a child and a spouse
    NOISE_STD=${NOISE_STD:-0.3}      # the noise of a node, as a fraction of its own signal's std;
                                     # higher than for a chain: a sharper Markov blanket
    EXP=${EXP:-chain_all}            # propagated and column-only perturbations
else
    TARGET=${TARGET:-last}           # target node: last, first or a node index
    NOISE_STD=${NOISE_STD:-0.1}
    EXP=${EXP:-chain}                # chain, or chain_local (no propagation)
fi

MODELS=${MODELS:-"tabfm_v1"}        # see MODELS_ALL in scripts/config.sh
                                     # every block of every model is read (--layer all)
PLOT_METRICS=${PLOT_METRICS:-"dcka ccos nmi"}   # one figure per metric (the ones fit for comparing layers)

# ---- Stages ------------------------------------------------------------------------------

cd "$(dirname "$0")/.." || exit 1            # the project root

# The generator's settings, and the names it gives the tables (default_name in
# dataset_generator.py, asked to the generator itself so they cannot drift apart).
GEN_FLAGS=(
    --structure "$STRUCTURE" --in_degree "$IN_DEGREE"
    --n_samples "$N_SAMPLES" --n_features "$N_FEATURES" --n_classes "$N_CLASSES"
    --target "$TARGET" --noise_std "$NOISE_STD" --noise_mode "$NOISE_MODE" --seeds $SEEDS
)
[ "${PERMUTE_COLUMNS:-1}" = 0 ] && GEN_FLAGS+=(--no_permute_columns)   # dag: columns in node order
[ "$STRUCTURE" = pool ] && GEN_FLAGS=(--pool "$POOL_SIZE")             # each table has its own settings
GEN_MODULE=dataset_generator.dataset_generator
if [ "$STRUCTURE" = groups ]; then
    GEN_MODULE=dataset_generator.cond_groups
    GEN_FLAGS=(--n_groups "$N_GROUPS" --n_samples "$N_SAMPLES")
fi
if [ "$STRUCTURE" = chamber ]; then
    GEN_MODULE=dataset_generator.causal_chambers
    GEN_FLAGS=(--experiment "$CHAMBER_EXPERIMENT" --targets $TARGETS --n_tables "$N_TABLES"
               --n_rows "$N_SAMPLES" --n_classes "$N_CLASSES")
fi
TABLES=" $(python -m "$GEN_MODULE" "${GEN_FLAGS[@]}" --names_only | tr '\n' ' ')"
# a group is run on its baseline: the experiment takes the other tables as contexts
RUN_TABLES="$TABLES"
[ "$STRUCTURE" = groups ] && RUN_TABLES=" $(echo $TABLES | tr ' ' '\n' | grep '_baseline$' | tr '\n' ' ')"

generate() {
    echo "== 1/3 Generate the tables"
    local graph=()
    [ "$STRUCTURE" = dag ] && graph=(--plot_graph)
    python -m "$GEN_MODULE" "${GEN_FLAGS[@]}" --skip_existing "${graph[@]}"
}

run() {
    if [ "$STRUCTURE" = chamber ]; then
        echo "== 2/3 Benchmark: models [$MODELS] and the classical detectors on every table at once"
        python -u benchmark_blanket.py --tables $TABLES --models $MODELS --n_jobs "$N_JOBS" \
            --tag "chamber_${CHAMBER_EXPERIMENT#uniform_}"
        return
    fi
    echo "== 2/3 Run the experiments: models [$MODELS], tables [$RUN_TABLES ]"
    # all_runs.sh loops over the layers of each model and skips the finished ones
    MODELS="$MODELS" DATASETS="$RUN_TABLES" EXP="$EXP" bash scripts/all_runs.sh
    accuracy
}

accuracy() {
    echo "== Clean test accuracy (saved in each run's json)"
    python3 utils/results.py accuracy --models $MODELS --datasets $RUN_TABLES --exp "$EXP"
}

plot() {
    echo "== 3/3 Plot"
    if [ "$STRUCTURE" = chamber ]; then
        cat "results/benchmark/blanket_chamber_${CHAMBER_EXPERIMENT#uniform_}_leaderboard.md" \
            || echo "No leaderboard yet: run the run stage first"
        return
    fi
    if [ "$STRUCTURE" = groups ]; then
        for model in $MODELS; do
            for metric in $PLOT_METRICS; do
                python compare_groups.py --model "$model" --metric "$metric" --exp "$EXP" --csv \
                    || echo "No plot for $model $metric"
            done
        done
        echo "Figures are in results/_fused/"
        return
    fi
    for model in $MODELS; do
        for table in $TABLES; do
            for metric in $PLOT_METRICS; do
                python fuse_plot.py --model "$model" --dataset "$table" --exp "$EXP" \
                    --metric "$metric" || echo "No plot for $model $table $metric"
                if [ "$STRUCTURE" != chain ]; then
                    python fuse_plot.py --model "$model" --dataset "$table" --exp "$EXP" \
                        --metric "$metric" --by_role || echo "No role plot for $model $table $metric"
                fi
            done
        done
    done
    echo "Figures are in results/_fused/"
}

case "${1:-all}" in
    generate) generate ;;
    run) run ;;
    plot) plot ;;
    accuracy) accuracy ;;
    all) generate && run && plot ;;
    *)
        echo "Usage: bash scripts/pipeline.sh [generate|run|plot|accuracy|all]" >&2
        exit 1
        ;;
esac
