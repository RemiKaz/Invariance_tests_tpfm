from models.limix_v2_model import LimiXV2
from models.mitra_v1_model import MitraV1
from models.mitra_v2_model import MitraV2
from models.tabfm_v1_model import TabFMV1
from models.tabicl_v1_model import TabICLV1
from models.tabicl_v2_model import TabICLV2
from models.tabpfn_v2_model import TabPFNV2
from models.tabpfn_v3_model import TabPFNV3

# One entry per model version, adding a version means one file and one line here.
MODELS = {
    "tabpfn_v3": TabPFNV3,
    "tabpfn_v2": TabPFNV2,
    "tabicl_v2": TabICLV2,
    "tabicl_v1": TabICLV1,
    "tabfm_v1": TabFMV1,
    "mitra_v1": MitraV1,
    "limix_v2": LimiXV2,
    "mitra_v2": MitraV2,
}
MODEL_NAMES = tuple(MODELS)


def model_importer(
    model_name,
    device="cuda",
    model_path=None,
    n_estimators=1,
    feature_slot="target",
    seed=0,
    dict_params=None,
):
    """Import the specified tabular foundation model.

    Args:
        model_name (str): The name of the model version to import, in MODEL_NAMES.
        device (str, optional): The device to run the model on. Defaults to "cuda".
        model_path (str, optional): The path of the checkpoint, None uses the per-model
            default. Defaults to None.
        n_estimators (int, optional): The size of the ensemble. Defaults to 1.
        feature_slot (str, optional): How per-cell activations are reduced to one vector
            per row, in [target, mean, flatten]. Defaults to "target".
        seed (int, optional): The random state of the classifier. Defaults to 0.
        dict_params (dict, optional): Extra parameters forwarded to the classifier,
            to do ablations. Defaults to None.

    Returns:
        BaseTabularPFM: The imported model.

    Raises:
        ValueError: If the model is not implemented.
    """
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not implemented ! Available: {list(MODEL_NAMES)}")
    print(f"Importing model {model_name} ...")
    return MODELS[model_name](
        model_path=model_path,
        n_estimators=n_estimators,
        seed=seed,
        dict_params=dict_params,
        device=device,
        feature_slot=feature_slot,
    )
