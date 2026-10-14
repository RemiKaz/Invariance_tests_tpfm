from abc import ABC, abstractmethod

import numpy as np

from utils.metrics import METRICS


class Experiment(ABC):
    """A sweep of perturbations of growing intensity applied to a dataset.

    An experiment only says which intensities are swept and how the dataset is
    perturbed at a given intensity: `run_experiment` does the fitting, the embedding
    and the comparison to the unperturbed baseline.

    Attributes:
        name (str): The short name of the experiment, used on the command line.
        title (str): The title of the plot panel.
        xlabel (str): The label of the x axis of the plot panel.
    """

    name = ""
    title = ""
    xlabel = ""

    @abstractmethod
    def levels(self, dataset, n_levels):
        """The perturbation intensities swept by the experiment.

        Args:
            dataset (TabularDataset): The unperturbed dataset.
            n_levels (int): The number of intensity steps asked for.

        Returns:
            np.ndarray: The intensities.
        """

    @abstractmethod
    def perturb(self, dataset, level, rng):
        """Apply the perturbation at one intensity.

        Args:
            dataset (TabularDataset): The unperturbed dataset.
            level (int): The intensity of the perturbation.
            rng (np.random.Generator): The random generator.

        Returns:
            TabularDataset: The perturbed dataset.
        """


def run_experiment(experiment, model, dataset, baseline, rng=None, n_levels=20, metrics=("cosine",)):
    """Run one experiment: similarity to the baseline vs. perturbation intensity, per layer.

    Fitting the model and reading its embeddings is what costs; comparing them to the
    baseline is cheap next to that. So every layer is read from the same forward pass
    (`model.get_embeddings_of`) and every metric asked for is computed on the same
    embeddings at each level, rather than rerunning the whole sweep per layer and metric.

    Args:
        experiment (Experiment): The experiment to run.
        model (BaseTabularPFM): The model whose embeddings are compared.
        dataset (TabularDataset): The unperturbed dataset.
        baseline (dict): layer -> the embeddings of the unperturbed dataset's test rows, the
            layers to read.
        rng (np.random.Generator, optional): The random generator. Defaults to None.
        n_levels (int, optional): The number of intensity steps. Defaults to 20.
        metrics (tuple, optional): Names of metrics in `utils.metrics.METRICS` used to
            compare the perturbed embeddings to the baseline. Defaults to ("cosine",).

    Returns:
        dict: layer -> metric name -> the experiment name, title, xlabel, levels and values,
            None at a level whose training labels hold a single class.
    """
    rng = np.random.default_rng() if rng is None else rng
    print(f"Running experiment {experiment.name} ...")
    layers = list(baseline)

    levels = np.asarray(experiment.levels(dataset, n_levels))
    values = {layer: {metric: [] for metric in metrics} for layer in layers}
    for level in levels:
        perturbed = experiment.perturb(dataset, int(level), rng)
        if len(np.unique(perturbed.y_train)) < 2:
            # The last steps of a row removal can leave a single class, which a classifier
            # cannot be fitted on (LimiX refuses it): the level is undefined for every model.
            print(f"  level {level}: a single class left in y_train, recorded as None")
            for layer in layers:
                for metric in metrics:
                    values[layer][metric].append(None)
            continue
        model.fit(perturbed.X_train, perturbed.y_train)
        embeds, errors = model.get_embeddings_of(perturbed.X_test, layers)
        if errors:
            layer, error = next(iter(errors.items()))
            raise RuntimeError(f"Layer {layer} failed at level {level}") from error
        for layer in layers:
            for metric in metrics:
                values[layer][metric].append(METRICS[metric].fn(embeds[layer], baseline[layer]))
        # one line per level, on the first layer read: the others are in the json
        print(
            f"  level {level} (layer {layers[0]}): "
            + ", ".join(f"{m}={values[layers[0]][m][-1]:.4f}" for m in metrics)
        )

    return {
        layer: {
            metric: {
                "name": experiment.name,
                "title": experiment.title,
                "xlabel": experiment.xlabel,
                "levels": levels.tolist(),
                "similarities": values[layer][metric],
            }
            for metric in metrics
        }
        for layer in layers
    }
