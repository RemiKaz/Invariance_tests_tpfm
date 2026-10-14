import json
import os
from dataclasses import dataclass

import numpy as np

from dataset_generator.scm_graph import node_roles
from datasets.custom_scm_dataset import CustomSCMDataset
from datasets.tabular_dataset import TabularDataset

DEFAULT_DATA_DIR = "/lustre/fswork/projects/rech/ycq/ufb58bn/DataRemi/causal_chamber_data"


def causal_chamber_data_dir():
    """Where the Causal Chamber tables live: $CAUSAL_CHAMBER_DATA, else the project's lustre folder."""
    return os.environ.get("CAUSAL_CHAMBER_DATA", DEFAULT_DATA_DIR)


def causal_chamber_paths(name, data_dir=None):
    """The (table, metadata) file paths of a stored Causal Chamber table."""
    data_dir = data_dir or causal_chamber_data_dir()
    return os.path.join(data_dir, f"{name}.npz"), os.path.join(data_dir, f"{name}.json")


@dataclass
class CausalChamberDataset(CustomSCMDataset):
    """A table of real measurements from a Causal Chamber (Gamella et al., Nature Machine
    Intelligence 2025), with the ground-truth causal graph of the chamber.

    It is stored like a table of `dataset_generator` (see `dataset_generator/causal_chambers.py`)
    and is one for every tool that only reads the nodes and the graph: the column-only
    perturbations (`chain_noise_local`, `chain_randomize_local`, the group `blanket`), the roles
    of `fuse_plot.py --by_role`, the Markov blanket benchmark. What it does not have is the
    structural equations: the system is physical, so a node cannot be changed and its
    descendants recomputed, and `noise` is None (the propagated experiments refuse it).

    The metadata holds two graphs over the same variables. `parents` is the chamber's ground
    truth as published, `effective_parents` keeps only the edges whose effect is measurable in
    the experiment the rows come from (see `edge_strength`): in the light tunnel, the LEDs next
    to the sensors (l_11 ... l_32) have a published edge into them but no visible effect at
    the ranges of the reference experiment. A method cannot be blamed for missing an edge that
    leaves no trace in the data, so the benchmark reports both truths.
    """

    @classmethod
    def load(cls, name, test_size=0.33, seed=42, data_dir=None):
        """Load a stored Causal Chamber table and split it in a train and a test set.

        Args:
            name (str): The name of the table (the file name without extension).
            test_size (float, optional): The test proportion. Defaults to 0.33.
            seed (int, optional): The seed of the split. Defaults to 42.
            data_dir (str, optional): The folder of the tables. Defaults to
                `causal_chamber_data_dir()`.

        Returns:
            CausalChamberDataset: The split dataset.
        """
        table_path, metadata_path = causal_chamber_paths(name, data_dir)
        if not (os.path.exists(table_path) and os.path.exists(metadata_path)):
            raise FileNotFoundError(f"No Causal Chamber table {name!r} in {os.path.dirname(table_path)}")
        with open(metadata_path) as file:
            metadata = json.load(file)
        table = np.load(table_path)
        base = TabularDataset.from_xy(name, table["X"], table["y"], test_size=test_size, seed=seed)
        return cls(**vars(base), metadata=metadata, nodes=table["nodes"], noise=None)

    @property
    def variables(self):
        """The chamber variable behind every node (red, ir_1, pol_1, ...)."""
        return self.metadata["variables"]

    @property
    def effective_roles(self):
        """The roles towards the target in the graph of the measurable edges only."""
        return node_roles(self.metadata["effective_parents"], self.target_node)
