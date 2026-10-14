#!/bin/bash
# Run every experiment on every dataset and model, every layer, one after the other. It is what
# job.slurm runs, and it can also be run by hand inside an allocation. MODELS / DATASETS
# restrict it, e.g. to the one (model, dataset) pair dev_watch.sh submitted a job for.
#
# Every (model, dataset) is one process reading all the layers of the model from the same
# forward passes. The pairs whose results are complete are skipped, and inside a pair only the
# missing (layer, metric) pairs are computed, so after a time-out `sbatch job.slurm` again
# carries on where it stopped. FORCE=1 redoes them.
#
# Results go to results/runs/<model>/<dataset>/<exp>.json, and the curves of all the layers
# of a model are put on one figure with:
#     python fuse_plot.py --model <model> --dataset <dataset> --exp all --metric ccos

cd "$(dirname "$0")/.." || exit 1            # the project root, where run_exps.py is
source scripts/config.sh

failed=0
while read -r model dataset; do
    run="${model}_${dataset}_${EXP}"
    echo "Running $run, metrics: $METRICS_ALL"
    # A failing run must not stop the others.
    if ! python -u run_exps.py --model "$model" --dataset "$dataset" --exp "$EXP" \
        --layer all --metric $METRICS_ALL ${FORCE:+--force}; then
        echo "FAILED: $run" >&2
        failed=$((failed + 1))
    fi
done < <(pending_pairs "${MODELS:-$MODELS_ALL}" "${DATASETS:-$DATASETS_ALL}")

echo "Done, $failed runs failed"
exit $((failed > 0))
