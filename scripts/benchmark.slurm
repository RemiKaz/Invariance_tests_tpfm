#!/bin/bash
#SBATCH --job-name=blanket_benchmark
#SBATCH --output=logs/%x_%j.out
#SBATCH --account=ycq@a100
#SBATCH -C a100
# The dev QOS only (2h max, starts within minutes).
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8           # the share of cores of one GPU on the A100 nodes
#SBATCH --hint=nomultithread
#SBATCH --time=00:58:00

# The whole Markov blanket benchmark in one job (benchmark_blanket.py): the classical detectors
# on the 8 cores, in parallel over the tables, then every model of MODELS on the GPU, loaded once
# for all the tables. The model scores are cached per table (results/blanket_scores/), so after
# a time-out `sbatch` again carries on where it stopped. Submit from the project root with the
# environment loaded (the job inherits it, as job.slurm does; the compute nodes are offline):
#     source $WORK/setup_env.sh
#     MODELS=tabpfn_v3 sbatch scripts/benchmark.slurm
#     MODELS=tabpfn_v3 LAYERS="final output" sbatch scripts/benchmark.slurm   # embedding and prediction
#     MODELS=tabpfn_v3 PROBE=default sbatch --time=01:30:00 scripts/benchmark.slurm   # + embedding variants (7 passes)
#     MODELS=tabpfn_v3 PROBE=noise TAG=chamber_ref_noise sbatch --time=02:00:00 scripts/benchmark.slurm   # noise sweep
#     TABLES='dagpool_*' MODELS="limix_v2" TAG=dagpool sbatch scripts/benchmark.slurm

cd "$SLURM_SUBMIT_DIR"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1   # one thread per table for the classical tests: the parallelism is over the tables

python -u benchmark_blanket.py --tables ${TABLES:-'cc_lt_ref_*'} --models ${MODELS:-tabpfn_v3} \
    --layers ${LAYERS:-final} ${PROBE:+--probe $PROBE} --n_jobs "$SLURM_CPUS_PER_TASK" --tag "${TAG:-chamber_ref}"
