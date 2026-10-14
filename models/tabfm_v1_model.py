import sys

from models.base_model import BaseTabularPFM, Readout


def _import_tabfm():
    """Import tabfm with its PyTorch backend only.

    tabfm tries its JAX backend first and only tolerates an ImportError. The flax of the
    IDRIS module is too old (no `nnx.dataclass`) and fails with an AttributeError instead,
    so flax is hidden while tabfm is imported, which makes that attempt fail cleanly.
    """
    missing = object()
    flax = sys.modules.get("flax", missing)
    sys.modules["flax"] = None  # `import flax` now raises ImportError
    try:
        import tabfm
    finally:
        if flax is missing:
            del sys.modules["flax"]
        else:
            sys.modules["flax"] = flax
    return tabfm


class TabFMV1(BaseTabularPFM):
    """TabFM v1.0.0 (Google), PyTorch backend, read with forward hooks.

    The forward pass alternates column and row stages before the ICL transformer:
    cell_embedder -> col_embedder -> row_interactor -> col_embedder_2 -> row_interactor_2
    -> icl_predictor. A bare `tf_col` or `tf_row` would not say which of the two stages it
    is, so every stage is numbered in forward order:

    - `tf_col_1`, `tf_col_2`: the output of the set transformer of the first / second
      column stage, one vector per table cell, before the linear map and norm that follow
      it. `--feature_slot` reduces it to one vector per row, "mean" (the default here) or
      "flatten"; "target" has no meaning here and is read as "mean". The CLS slots and
      the padded columns are dropped.
    - `tf_row_1`, `tf_row_2`: the CLS tokens after the first / second row transformer,
      concatenated. The second is what the ICL transformer reads.
    - `tf_icl` (the default, also `--layer last`): the end of the ICL transformer, after
      its final norm, what the decoder reads.

    The blocks inside the four column and row stages can be read too: `col1_<i>` and
    `col2_<i>` (block `i` of the set transformer of column stage 1 / 2, read like `tf_col_1`
    / `tf_col_2`, the last one being that stage's own output), `row1_<i>` and `row2_<i>`
    (block `i` of the row transformer of row stage 1 / 2, the CLS tokens of its output,
    before the stage's final norm). The blocks of the ICL transformer are `--layer <index>`.

    Only the first ensemble member is read, `n_estimators=1` keeps it the only one.
    """

    LAYER_CONTAINERS = ("icl_predictor.tf_icl.blocks",)
    LAYER_ALIASES = {
        "tf_col_1": "col_embedder.tf_col",
        "tf_row_1": "row_interactor",
        "tf_col_2": "col_embedder_2.tf_col",
        "tf_row_2": "row_interactor_2",
        "tf_icl": "icl_predictor.ln",
    }
    # The set transformers of the column stages, read by `_tf_col_readout`.
    COL_STAGES = ("col_embedder.tf_col", "col_embedder_2.tf_col")

    def _build_classifier(self):
        tabfm = _import_tabfm()

        network = tabfm.tabfm_v1_0_0_pytorch.load(
            model_type="classification", checkpoint_path=self.model_path, device=self.device
        )
        return tabfm.TabFMClassifier(
            model=network,
            n_estimators=self.n_estimators,
            random_state=self.seed,
            **self.dict_params,
        )

    def _torch_model(self):
        return self.clf.model

    @staticmethod
    def _blocks(kind, stage):
        """The dotted name of the blocks of column (`col`) or row (`row`) stage 1 or 2."""
        suffix = "" if stage == 1 else "_2"
        if kind == "col":
            return f"col_embedder{suffix}.tf_col.blocks"
        return f"row_interactor{suffix}.tf_row.blocks"

    def default_layers(self):
        model = self._torch_model()
        layers = []
        for stage in (1, 2):
            n_col = len(model.get_submodule(self._blocks("col", stage)))
            n_row = len(model.get_submodule(self._blocks("row", stage)))
            # the last column block is the output of tf_col, it is not listed twice
            layers += [f"col{stage}_{i}" for i in range(n_col - 1)] + [f"tf_col_{stage}"]
            layers += [f"row{stage}_{i}" for i in range(n_row)] + [f"tf_row_{stage}"]
        return layers + self.stack_layers() + ["tf_icl"]

    def _named_readouts(self):
        model = self._torch_model()
        named = {"last": self._module_readout(self.LAYER_ALIASES["tf_icl"])}
        for stage in (1, 2):
            for kind in ("col", "row"):
                blocks = self._blocks(kind, stage)
                for i in range(len(model.get_submodule(blocks))):
                    if kind == "col":
                        readout = self._tf_col_readout(f"{blocks}.{i}")
                    else:
                        readout = self._row_block_readout(f"{blocks}.{i}", blocks.split(".")[0])
                    named[f"{kind}{stage}_{i}"] = readout
        return named

    def _module_readout(self, name):
        if name in self.COL_STAGES:
            return self._tf_col_readout(name)
        if name == self.LAYER_ALIASES["tf_row_1"]:
            # The first row stage outputs the CLS tokens and the features, keep the CLS
            # tokens: (batch, rows, cls + features, dim) -> (batch, rows, cls * dim).
            n_cls = self._torch_model().cls_tokens.shape[0]
            return Readout(name, lambda capture: capture.chunks[-1][..., :n_cls, :].flatten(start_dim=-2))
        return super()._module_readout(name)

    def _row_block_readout(self, name, stage_name):
        """Read the CLS tokens after one block of a row transformer.

        A block sees every row as its own sequence of (CLS + feature) tokens, so its output is
        shaped (rows of the table, tokens, dim) and the row transformer may process it in
        chunks. The chunks of the last forward pass are put back together, and the shape of
        the whole stage's output tells how the rows split into tables.
        """
        import torch

        n_cls = self._torch_model().cls_tokens.shape[0]

        def reduce(capture):
            n_tables, n_rows = capture.watched["stage"][:2]
            tokens = torch.cat(capture.chunks)  # (tables * rows, tokens, dim)
            if tokens.shape[0] != n_tables * n_rows:
                raise RuntimeError(
                    f"Captured {tokens.shape[0]} row sequences from {name}, expected "
                    f"{n_tables * n_rows} ({n_tables} tables x {n_rows} rows)"
                )
            activations = tokens.reshape(n_tables, n_rows, *tokens.shape[1:])[..., :n_cls, :]
            return activations.flatten(start_dim=-2)  # the CLS tokens, concatenated

        return Readout(name, reduce, end="", watch={"stage": stage_name})

    def _tf_col_readout(self, name):
        """Read a column-wise set transformer (`col_embedder.tf_col`) or one of its blocks
        (`col_embedder.tf_col.blocks.<i>`), which output the same layout.

        The columns are processed in chunks, which call the set transformer once per chunk,
        so the calls of the last forward pass are gathered and put back together.
        """
        import torch

        stage_name = name.split(".")[0]
        # The second column stage also sees the CLS slots, the first one does not.
        n_reserved = self._torch_model().cls_tokens.shape[0] if stage_name == "col_embedder_2" else 0

        def reduce(capture):
            n_tables, _, n_columns, _ = capture.watched["stage"]
            n_features = capture.end_kwargs.get("d")
            # Each call is shaped (columns of the chunk, rows, dim), the columns of all the
            # tables being stacked table after table.
            columns = torch.cat(capture.chunks)
            if columns.shape[0] != n_tables * n_columns:
                raise RuntimeError(
                    f"Captured {columns.shape[0]} column embeddings from {name}, expected "
                    f"{n_tables * n_columns} ({n_tables} tables x {n_columns} columns)"
                )
            # First table only: (columns, rows, dim) -> (1, rows, features, dim), leaving out
            # the CLS slots and the columns padded past the active features.
            n_active = int(n_features[0]) if n_features is not None else n_columns - n_reserved
            columns = columns.reshape(n_tables, n_columns, *columns.shape[-2:])[0]
            activations = columns[n_reserved : n_reserved + n_active].transpose(0, 1).unsqueeze(0)
            return self._cells_without_target(activations)

        return Readout(name, reduce, end="", watch={"stage": stage_name})
