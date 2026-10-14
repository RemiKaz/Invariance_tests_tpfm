import os

os.environ.setdefault("HF_HOME", os.path.join(os.environ["WORK"], "hf_cache"))

from huggingface_hub import hf_hub_download, snapshot_download

CHECKPOINTS = (
    ("Prior-Labs/tabpfn_3", "tabpfn-v3-classifier-v3_default.ckpt"),  # --model tabpfn_v3
    ("Prior-Labs/TabPFN-v2-clf", "tabpfn-v2-classifier.ckpt"),  # --model tabpfn_v2
    ("jingang/TabICL", "tabicl-classifier-v2-20260212.ckpt"),  # --model tabicl_v2
    ("jingang/TabICL", "tabicl-classifier-v1-20250208.ckpt"),  # --model tabicl_v1
    ("stable-ai/LimiX-2", "LimiX-2.ckpt"),  # --model limix_v2 (1.6 GB, non-commercial licence)
)

for repo_id, filename in CHECKPOINTS:
    print(hf_hub_download(repo_id=repo_id, filename=filename))

# --model tabfm_v1: the classification weights are a folder (config.json + safetensors),
# under the same cache layout as tabfm's own loader uses, so it finds them offline. The
# root config.json is what the Hugging Face mixin looks for first.
print(
    snapshot_download(
        repo_id="google/tabfm-1.0.0-pytorch", allow_patterns=["config.json", "classification/**"]
    )
)

# --model mitra_v1 / mitra_v2: a repo of two files (config.json + model.safetensors), read
# by repo id, so the whole repo goes to the cache.
for repo_id in ("autogluon/mitra-classifier", "autogluon/mitra-classifier-2"):
    print(snapshot_download(repo_id=repo_id))

# Datasets downloaded from OpenML (--dataset phoneme): cached as .npz in $WORK/scikit_learn_data,
# which is all the compute nodes read.
from datasets.dataset_importer import SKLEARN_LOADERS

for name in ("phoneme", "spambase"):
    X, y = SKLEARN_LOADERS[name]()
    print(f"{name}: {X.shape[0]} rows, {X.shape[1]} features cached")
