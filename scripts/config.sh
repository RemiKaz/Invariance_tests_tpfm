# Shared defaults of the campaign scripts (all_runs.sh, dev_watch.sh, pipeline.sh).
# Source it, don't execute it. Each var keeps its value if the caller already set one, so
# `MODELS_ALL="mitra_v2" bash scripts/dev_watch.sh` overrides it everywhere downstream.

DATASETS_ALL=${DATASETS_ALL:-"breast_cancer iris wine digits phoneme spambase"}      # datasets/dataset_importer.py
MODELS_ALL=${MODELS_ALL:-"tabpfn_v3 tabpfn_v2 tabicl_v2 tabicl_v1 tabfm_v1 mitra_v2 mitra_v1 limix_v2"}
# The metrics computed for every run (utils/metrics.py). They all come from the same
# embeddings, so a layer is one sweep whatever the number of metrics; a rerun only computes
# the (layer, metric) pairs still missing. To compare layers of one model on one dataset, use
# ccos and nmi: in synthetic checks they kept the right order between layers of different width
# and shape every time, while plain cosine inverts it when a layer has a large shared mean
# vector and CKA when layers differ in spectrum. dcka is kept for the geometry view along one
# curve, where every metric orders levels correctly.
METRICS_ALL=${METRICS_ALL:-"cosine ccos nmi dcka"}
EXP=${EXP:-all}                              # "all" is every experiment, see experiments/
# Every run reads every layer of its model (run_exps.py --layer all, the list is the model's
# default_layers()).

# Every (model, dataset) pair of $1 x $2 (default: the *_ALL lists), one "model dataset" per line.
all_pairs() {
    local model dataset
    for model in ${1:-$MODELS_ALL}; do
        for dataset in ${2:-$DATASETS_ALL}; do
            echo "$model $dataset"
        done
    done
}

# The pairs whose run of $EXP is not complete yet (a --layer all run saved with every metric of
# METRICS_ALL for every layer), read from the result files; every pair with FORCE=1. Only needs
# python3 and the standard library, so it runs on a login node. Run from the project root.
pending_pairs() {
    if [ -n "$FORCE" ]; then
        all_pairs "$@"
    else
        python3 utils/results.py pending --models ${1:-$MODELS_ALL} --datasets ${2:-$DATASETS_ALL} \
            --exp "$EXP" --metrics $METRICS_ALL
    fi
}
