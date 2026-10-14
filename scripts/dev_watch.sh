#!/bin/bash
# Keep the dev queue full until the campaign is done, and report the progress: every INTERVAL
# seconds it looks at `squeue -u $USER`, submits one job.slurm per (model, dataset) pair that is
# neither finished, queued nor running - as many as qos_gpu_a100-dev's 10 job slots allow - and
# prints one line: pairs done / running / pending / still to submit.
#
# Run it on a login node, in the foreground or in tmux / screen (Ctrl-C stops the watching, the
# jobs stay queued). What is finished is read from the result files (pending_pairs in config.sh).
#
#     EXP=chain_all MODELS_ALL="mitra_v2" DATASETS_ALL="$(printf 'dagpool_%02d ' $(seq 0 19))" \
#         bash scripts/dev_watch.sh
#     INTERVAL=60 MAX_TRIES=2 ... bash scripts/dev_watch.sh      # poll every minute, 2 tries a pair
#     DRY=1 ... bash scripts/dev_watch.sh                        # one round, submits nothing
#
# A pair whose job ended without all its results (killed at the 58 minutes of job.slurm, or a
# crash) is submitted again, at most MAX_TRIES times: past that it is reported and left alone,
# its logs are in logs/<model>_<dataset>_<jobid>.out.

INTERVAL=${INTERVAL:-120}       # seconds between two looks at the queue
MAX_TRIES=${MAX_TRIES:-3}       # submissions of one pair, the first one included
MAX_SUBMIT=${MAX_SUBMIT:-10}    # qos_gpu_a100-dev's MaxSubmitPU, running and pending together
QOS=${QOS:-qos_gpu_a100-dev}

cd "$(dirname "$0")/.." || exit 1
source scripts/config.sh
mkdir -p logs

declare -A tries
total=$(all_pairs | wc -l)
[ "$total" -eq 0 ] && { echo "No (model, dataset) pair, check MODELS_ALL and DATASETS_ALL"; exit 1; }

while true; do
    # job name -> "STATE elapsed", the names being <model>_<dataset> as set at submission below
    declare -A live=()
    while IFS='|' read -r name state elapsed; do
        [ -n "$name" ] && live[$name]="$state $elapsed"
    done < <(squeue -h -u "$USER" -o "%j|%T|%M" 2>/dev/null)
    in_flight=$(squeue -h -u "$USER" --qos="$QOS" 2>/dev/null | wc -l)

    mapfile -t pairs < <(pending_pairs)
    n_done=$((total - ${#pairs[@]})); n_running=0; n_pending=0
    running_list=""; gave_up=(); todo=()
    for pair in "${pairs[@]}"; do
        read -r model dataset <<<"$pair"
        name="${model}_${dataset}"
        if [ -n "${live[$name]}" ]; then
            read -r state elapsed <<<"${live[$name]}"
            if [ "$state" = RUNNING ]; then
                n_running=$((n_running + 1)); running_list="$running_list $name($elapsed)"
            else
                n_pending=$((n_pending + 1))
            fi
        elif [ "${tries[$name]:-0}" -ge "$MAX_TRIES" ]; then
            gave_up+=("$name")
        else
            todo+=("$pair")
        fi
    done

    submitted=0
    slots=$((MAX_SUBMIT - in_flight))
    for pair in "${todo[@]}"; do
        [ "$submitted" -ge "$slots" ] && break
        read -r model dataset <<<"$pair"
        name="${model}_${dataset}"
        if [ -n "$DRY" ]; then
            echo "would submit $name"
        else
            MODELS=$model DATASETS=$dataset sbatch --job-name="$name" job.slurm >/dev/null || break
            if [ "${tries[$name]:-0}" -ge 1 ]; then
                echo "  $name ended without all its results, submitted again ($((${tries[$name]} + 1))/$MAX_TRIES)"
            fi
        fi
        tries[$name]=$(( ${tries[$name]:-0} + 1 ))
        submitted=$((submitted + 1))
    done

    left=$((${#todo[@]} - submitted))
    printf '[%s] done %d/%d | running %d | pending %d | to submit %d (submitted now: %d)\n' \
        "$(date +%H:%M:%S)" "$n_done" "$total" "$n_running" "$n_pending" "$left" "$submitted"
    [ -n "$running_list" ] && echo "  running:$running_list"
    [ ${#gave_up[@]} -gt 0 ] && echo "  gave up after $MAX_TRIES tries: ${gave_up[*]}"

    if [ "$n_done" -eq "$total" ]; then
        echo "All $total pairs are done"
        exit 0
    fi
    if [ "$n_running" -eq 0 ] && [ "$n_pending" -eq 0 ] && [ "$submitted" -eq 0 ] && [ "$left" -eq 0 ]; then
        echo "Nothing left to run, but ${#gave_up[@]} pair(s) never finished in $MAX_TRIES tries: ${gave_up[*]}"
        echo "See logs/<name>_<jobid>.out, then rerun (a rerun starts the counting again)"
        exit 1
    fi
    [ -n "$DRY" ] && exit 0
    sleep "$INTERVAL"
done
