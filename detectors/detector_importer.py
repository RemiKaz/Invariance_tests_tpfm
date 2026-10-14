from detectors.grow_shrink import GrowShrink
from detectors.iamb import IAMB
from detectors.marginal import Marginal
from detectors.pfm_perturbation import PFMPerturbation
from detectors.pfm_probe import PFMProbe
from detectors.total_conditioning import TotalConditioning

DETECTORS = {
    "iamb": IAMB,
    "grow_shrink": GrowShrink,
    "total_conditioning": TotalConditioning,
    "marginal": Marginal,
    "pfm": PFMPerturbation,
    "probe": PFMProbe,
}


def detector_importer(detector_name, dict_params=None):
    """Import a Markov blanket detector.

    Args:
        detector_name (str): A key of DETECTORS. The classical ones take `test` (fisher_z,
            logistic, spline) and `alpha`, `pfm` takes `model`, `exp`, `metric`, `layer` and
            `results_dir`, `probe` takes `model`, `perturbations`, `seeds`, `readout`, `rule`, ...
        dict_params (dict, optional): The parameters of the detector. Defaults to None.

    Returns:
        BlanketDetector: The detector.

    Raises:
        ValueError: If the detector is not implemented.
    """
    dict_params = {} if dict_params is None else dict_params
    if detector_name not in DETECTORS:
        raise ValueError(f"Detector {detector_name} not implemented ! Available: {list(DETECTORS)}")
    return DETECTORS[detector_name](**dict_params)
