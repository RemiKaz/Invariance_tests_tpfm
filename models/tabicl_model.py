from models.base_model import BaseTabularPFM, Readout


class TabICL(BaseTabularPFM):
    """Shared TabICL wrapper, whose row representations only exist inside the forward pass.

    The versions live in tabicl_v1_model.py and tabicl_v2_model.py.

    Three representations are read with a forward hook, named as in the TabICL papers
    (the stages are the same in v1 and v2):

    - `tf_col`: the output of the column-wise set transformer, one vector per table cell,
      shaped (rows, features, dim). It is read before the affine map that turns it into
      the cell embeddings fed to the row transformer. `--feature_slot` reduces it to one
      vector per row, "mean" (the default here) or "flatten"; "target" has no meaning
      here, since TabICL has no target cell, and is read as "mean".
    - `tf_row`: the output of the row-wise transformer, the concatenated CLS tokens, one
      vector of 4 x dim per row.
    - `tf_icl` (the default, also `--layer last`): the end of the in-context-learning
      transformer, i.e. its output after the final LayerNorm, which is what the decoder
      reads to predict. One vector per row.

    Every block of the three stages can be read by its index, `col_<i>` (of tf_col, read like
    `tf_col`), `row_<i>` (of tf_row, the CLS-token slots read like `tf_row`: every block writes
    into the same reserved slots, so a block gives an earlier snapshot of the same
    representation) and `icl_<i>` (of the ICL transformer, read like `tf_icl`). The last block of
    tf_row does the CLS-only pooling and is `tf_row` itself. `tf_col_early` / `tf_col_middle` and
    `tf_row_early` / `tf_row_middle` name the blocks closing the first quarter / half of a stage.
    """

    LAYER_CONTAINERS = (
        "row_interactor.layers",
        "row_interactor",
        "tf_icl.layers",
        "icl_predictor.layers",
    )
    LAYER_ALIASES = {"tf_row": "row_interactor", "tf_col": "col_embedder.tf_col"}
    TF_COL = "col_embedder.tf_col"
    # Fractional depths of tf_col and tf_row, read like their stage's own final output.
    STAGE_DEPTHS = {"early": 0.25, "middle": 0.5}
    # The checkpoint TabICL downloads when no model_path is given, None keeps the library
    # default. Set by the version subclasses to pin the version.
    CHECKPOINT_VERSION = None

    def _build_classifier(self):
        from tabicl import TabICLClassifier

        params = dict(self.dict_params)
        if self.CHECKPOINT_VERSION is not None:
            params.setdefault("checkpoint_version", self.CHECKPOINT_VERSION)
        return TabICLClassifier(
            model_path=self.model_path,
            allow_auto_download=self.model_path is None,
            n_estimators=self.n_estimators,
            device=self.device,
            random_state=self.seed,
            **params,
        )

    def _n_blocks(self):
        """The number of readable blocks of the col, row and icl stages."""
        model = self._torch_model()
        return {
            "col": len(model.col_embedder.tf_col.blocks),
            # the last block of the row stage does the CLS-only pooling, it is `tf_row` itself
            "row": len(model.row_interactor.tf_row.blocks) - 1,
            "icl": len(model.icl_predictor.tf_icl.blocks),
        }

    def default_layers(self):
        n = self._n_blocks()
        # the last column block is the output of tf_col, it is not listed twice
        return (
            [f"col_{i}" for i in range(n["col"] - 1)]
            + ["tf_col"]
            + [f"row_{i}" for i in range(n["row"])]
            + ["tf_row"]
            + [f"icl_{i}" for i in range(n["icl"])]
            + ["tf_icl"]
        )

    def _named_readouts(self):
        model = self._torch_model()
        # The final LayerNorm only exists for pre-norm models, otherwise TF_icl ends itself.
        tf_icl = "icl_predictor.ln" if hasattr(model.icl_predictor, "ln") else "icl_predictor.tf_icl"
        n = self._n_blocks()
        named = {"tf_icl": self._module_readout(tf_icl)}
        named["last"] = named["tf_icl"]
        named.update({f"col_{i}": self._tf_col_readout(f"{self.TF_COL}.blocks.{i}") for i in range(n["col"])})
        named.update({f"row_{i}": self._row_block_readout(i) for i in range(n["row"])})
        named.update(
            {f"icl_{i}": self._module_readout(f"icl_predictor.tf_icl.blocks.{i}") for i in range(n["icl"])}
        )
        for depth, fraction in self.STAGE_DEPTHS.items():
            named[f"tf_col_{depth}"] = named[f"col_{self._depth_index(n['col'], fraction)}"]
            if n["row"] > 0:
                named[f"tf_row_{depth}"] = named[f"row_{self._depth_index(n['row'], fraction)}"]
        return named

    def _module_readout(self, name):
        if name == self.TF_COL:
            return self._tf_col_readout(name)
        return super()._module_readout(name)

    def _tf_col_readout(self, module):
        """Read the column-wise stage `col_embedder.tf_col`, or one of its blocks.

        The inference manager may split the columns over several calls of that submodule, and
        a prediction may run several forward passes (one per ensemble batch): the output of
        `col_embedder` closes a pass, its shape says how the calls of the last one fit
        together.
        """
        import torch

        col_embedder = self._torch_model().col_embedder

        def reduce(capture):
            n_tables, _, n_columns, _ = capture.end_shape
            # Each call is shaped (..., rows, dim), the leading dims index the columns.
            columns = torch.cat([chunk.reshape(-1, *chunk.shape[-2:]) for chunk in capture.chunks])
            if columns.shape[0] == n_tables * n_columns:
                columns = columns.reshape(n_tables, n_columns, *columns.shape[-2:])
            elif columns.shape[0] == n_columns:
                # Feature-shuffled ensembles embed the columns once and reorder them per table.
                columns = columns.unsqueeze(0)
            else:
                raise RuntimeError(
                    f"Captured {columns.shape[0]} column embeddings from {module}, "
                    f"expected {n_tables * n_columns} ({n_tables} tables x {n_columns} columns)"
                )
            # (tables, reserved + features, rows, dim) -> (tables, rows, features, dim), where
            # the first columns are reserved for the CLS tokens and carry no feature.
            activations = columns[:, col_embedder.reserve_cls_tokens :].transpose(1, 2)
            return self._cells_without_target(activations)

        return Readout(module, reduce, end="col_embedder")

    def _row_block_readout(self, index):
        """Read the CLS-token slots after one (non-final) block of the row transformer.

        Long tables may be processed in row chunks; the output of `row_interactor` closes a
        pass, its own hook firing after every nested block hook it triggered.
        """
        import torch

        n_cls = self._torch_model().row_interactor.num_cls

        def reduce(capture):
            activations = torch.cat(capture.chunks, dim=1)  # (tables, rows, cls + features, dim)
            return activations[..., :n_cls, :].flatten(start_dim=-2)  # the CLS tokens

        return Readout(f"row_interactor.tf_row.blocks.{index}", reduce, end="row_interactor")
