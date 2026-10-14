from models.base_model import Readout
from models.tabpfn_model import TabPFN, find_hf_checkpoint


class TabPFNV3(TabPFN):
    """TabPFN v3, whose stages mirror the ones of TabICL and are read with a forward hook.

    - `tf_col`: the feature distribution embedder, one vector per table cell, shaped
      (rows, feature groups, dim). `--feature_slot` reduces it to one vector per row,
      "mean" (the default here) or "flatten"; "target" has no meaning here and is read
      as "mean".
    - `tf_row`: the column aggregator, the CLS tokens concatenated as the model does
      before the ICL stage, one vector per row.
    - `tf_icl`: the end of the ICL transformer, after its output norm, what the decoder
      reads. One vector per row.

    The blocks inside the two first stages can be read too: `col_<i>` (layer `i` of
    `feature_distribution_embedder`, read like `tf_col`, its last layer being `tf_col`
    itself) and `row_<i>` (block `i` of `column_aggregator`, the CLS tokens read like
    `tf_row`; its last block does the CLS-only readout and is called directly, so it is
    `tf_row` itself). The blocks of the ICL transformer are `--layer <index>`.
    """

    # The main (in-context) transformer, addressed by `--layer <index>`. The column
    # aggregator and the feature embedder have their own stacks, reachable by name.
    LAYER_CONTAINERS = ("icl_blocks",) + TabPFN.LAYER_CONTAINERS
    # TABPFN_V3_CKPT overrides the path.
    DEFAULT_MODEL_PATH = find_hf_checkpoint(
        "Prior-Labs/tabpfn_3", "tabpfn-v3-classifier-v3_default.ckpt", "TABPFN_V3_CKPT"
    )
    STAGES = {
        "tf_col": "feature_distribution_embedder",
        "tf_row": "column_aggregator",
        "tf_icl": "output_norm",
    }
    LAYER_ALIASES = STAGES

    def _n_blocks(self):
        model = self._torch_model()
        return {
            "col": len(model.feature_distribution_embedder.layers),
            "row": len(model.column_aggregator.blocks) - 1,  # the last one is tf_row
        }

    def default_layers(self):
        n = self._n_blocks()
        # the last layer of the distribution embedder is tf_col, it is not listed twice
        return (
            [f"col_{i}" for i in range(n["col"] - 1)]
            + ["tf_col"]
            + [f"row_{i}" for i in range(n["row"])]
            + ["tf_row"]
            + self.stack_layers()
            + ["tf_icl", "last"]
        )

    def _named_readouts(self):
        n = self._n_blocks()
        named = super()._named_readouts()
        named.update(
            {f"col_{i}": self._stage_readout(f"feature_distribution_embedder.layers.{i}", "col") for i in range(n["col"])}
        )
        named.update({f"row_{i}": self._stage_readout(f"column_aggregator.blocks.{i}", "row_block") for i in range(n["row"])})
        return named

    def _module_readout(self, name):
        if name in self.STAGES.values():
            return self._stage_readout(name, {self.STAGES["tf_col"]: "col", self.STAGES["tf_row"]: "row"}.get(name))
        return super()._module_readout(name)

    def _stage_readout(self, name, kind):
        """Read one stage, or a block inside one.

        `kind` says how the captured activations become one vector per row: "col" (the cells,
        averaged or flattened), "row" (the CLS tokens of the aggregator's output, concatenated),
        "row_block" (the CLS tokens of a block inside the aggregator, which also holds the
        feature tokens) or None (as they are).

        Long tables are processed in row chunks, which call the stage once per chunk, so the
        calls of the last forward pass are gathered and put back together.
        """
        import torch

        n_cls = self._torch_model().column_aggregator.num_cls_tokens  # the CLS tokens come first

        def reduce(capture):
            # (tables, rows, ...): the row chunks are put back in order along the rows.
            activations = torch.cat(capture.chunks, dim=1)
            if kind == "col":
                return self._cells_without_target(activations)
            if kind == "row":
                return activations.flatten(start_dim=-2)  # the CLS tokens, concatenated
            if kind == "row_block":
                return activations[..., :n_cls, :].flatten(start_dim=-2)
            return self._cells(activations)

        return Readout(name, reduce, end="")
