#!/bin/bash
#SBATCH --job-name=sae_tabpfm
#SBATCH --output=logs/%x_%j.out
# No --error: stderr (tracebacks, FAILED lines) goes to the --output file, in order.
#SBATCH --account=ycq@a100          # `idrproj` lists the accounts, suffix = GPU partition
#SBATCH -C a100                     # the A100 nodes
# The dev QOS only: its jobs start within minutes, where the queue of the t3 QOS takes far
# longer than a run. It allows 2h and MaxSubmitPU=10 (jobs submitted at once, running and
# pending together), which scripts/dev_watch.sh respects to run a whole campaign.
#SBATCH --qos=qos_gpu_a100-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8           # the share of cores of one GPU on the A100 nodes
#SBATCH --hint=nomultithread
# Under the 2h of the dev QOS, which backfills sooner. A run killed by the time limit saves
# nothing, so a resubmission starts that (model, dataset) again, and skips the ones done.
#SBATCH --time=00:58:00

# Runs every experiment on every pending (model, dataset) pair, every layer, see
# scripts/all_runs.sh. One job is for a single pair or a small subset that fits in the time
# limit; scripts/dev_watch.sh submits one job per pair for a whole campaign.
#     mkdir -p logs        # once, Slurm opens the output files itself
#     MODELS="mitra_v2" DATASETS="iris" sbatch job.slurm
# For a V100 node instead, override the settings when submitting (10 cores per GPU):
#     sbatch --account=ycq@v100 -C v100 --qos=qos_gpu-dev --cpus-per-task=10 job.slurm

cd "$SLURM_SUBMIT_DIR"

# The A100 nodes take the modules of their own architecture, chosen from the account.
case "$SLURM_JOB_ACCOUNT" in
    *@a100) export JZ_ARCH=a100 ;;
esac

source setup_env.sh
nvidia-smi --query-gpu=name --format=csv,noheader  # the GPU the results come from

# The compute nodes have no internet access, the checkpoints have to be in $HF_HOME
# already. Run `python down_weight.py` on a login node if they are not.
export HF_HUB_OFFLINE=1

bash scripts/${SCRIPT:-all_runs.sh}   # SCRIPT="pipeline.sh run" runs one stage of the pipeline
