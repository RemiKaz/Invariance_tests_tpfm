import numpy as np

from dataset_generator.cond_groups import VARIANTS, group_names
from datasets.custom_scm_dataset import CustomSCMDataset
from experiments.base_experiment import Experiment


class ContextSwap(Experiment):
    """Same query rows, the context (the training table) taken from another table of the group.

    A table of a group (dataset_generator/cond_groups.py) is run as the baseline: its test rows
    are the queries, and the level says which table of the group provides the context the model
    is fitted on: 0 is the baseline's own (the similarity is 1), then `markov_equiv`, `same_cond`,
    `anticausal` and `near_graph`. The queries do not change, so the embeddings of the baseline
    and of the swapped context are aligned row by row and every similarity metric applies. The
    tables are independent samples with the same variable in the same column.

    What it measures is how much the embedding of a fixed query depends on the context it is
    read in: if the model follows P(y | X, context), contexts that share the baseline's P(y | X)
    (`markov_equiv`, `anticausal`, `same_cond`) should leave it alone, up to the sampling noise of
    the context (which `markov_equiv` and `anticausal`, samples of the baseline's own
    distribution, show), and `near_graph`, whose P(y | X) differs, should not.
    """

    name = "context_swap"
    title = "Same query rows, context from another table of the group"
    xlabel = "Context table: " + ", ".join(f"{i} {v}" for i, v in enumerate(VARIANTS))

    def levels(self, dataset, n_levels):
        if not isinstance(dataset, CustomSCMDataset) or "group" not in dataset.metadata:
            raise TypeError(f"{self.name} needs a table of a group stored by dataset_generator.cond_groups")
        return np.arange(len(VARIANTS))

    def perturb(self, dataset, level, rng):
        name = group_names(dataset.metadata["group"]["index"])[VARIANTS[level]]
        if name == dataset.name:
            return dataset
        context = CustomSCMDataset.load(name, test_size=dataset.test_size, seed=dataset.seed)
        return dataset.perturbed(X_train=context.X_train, y_train=context.y_train)
