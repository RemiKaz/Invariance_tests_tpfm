"""A pool of diverse DAG tables: every table has its own configuration, drawn from a prior over
configurations (in the spirit of TabICL's hyperparameter sampling), so that a study is not
about a single graph shape.

What varies from one table to the next: the number of features, the number of parents of a node
(fixed, or drawn per node), the number of roots, the activation, the noise level, the root
distribution and the number of classes. The tables are named `dagpool_00`, `dagpool_01`, ... and
each one is reproducible from its index: `sample_pool_config(i)` is always the same.
"""

import numpy as np

from dataset_generator.dataset_generator import PriorConfig

POOL_SEED = 1000

# The choices of the prior. Several tanh entries: it is the mildest activation, drawn more often.
N_FEATURES = (6, 8, 10, 12)
IN_DEGREES = (2, 3)  # 1 would be a tree: no spouse, so no target with a full Markov blanket
IN_DEGREE_MODES = ("fixed", "random")
N_ROOTS = (1, 2, 3)
ACTIVATIONS = ("tanh", "tanh", "relu", "leaky_relu", "sin", "random")
NOISE_STDS = (0.1, 0.3, 0.5)
N_CLASSES = (2, 3, 4)
ROOT_SAMPLINGS = ("normal", "uniform")


def pool_name(index):
    """The name of the table `index` of the pool."""
    return f"dagpool_{index:02d}"


def sample_pool_config(index, pool_seed=POOL_SEED, attempt=0):
    """The configuration of table `index` of the pool.

    Args:
        index (int): The index of the table.
        pool_seed (int, optional): The seed of the pool. Defaults to POOL_SEED.
        attempt (int, optional): Bumped to draw another configuration for the same index, when
            the first one has no node with a full Markov blanket. Defaults to 0.

    Returns:
        PriorConfig: The configuration, its `seed` being the seed of the table.
    """
    rng = np.random.default_rng([pool_seed, index, attempt])
    return PriorConfig(
        n_samples=1000,
        n_features=int(rng.choice(N_FEATURES)),
        n_classes=int(rng.choice(N_CLASSES)),
        target="blanket",
        activation=str(rng.choice(ACTIVATIONS)),
        weight_std=1.0,
        noise_std=float(rng.choice(NOISE_STDS)),
        noise_mode="relative",
        root_sampling=str(rng.choice(ROOT_SAMPLINGS)),
        seed=int(rng.integers(0, 2**31 - 1)),
        structure="dag",
        in_degree=int(rng.choice(IN_DEGREES)),
        in_degree_mode=str(rng.choice(IN_DEGREE_MODES)),
        n_roots=int(rng.choice(N_ROOTS)),
    )
