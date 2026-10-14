from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Capture:
    """What the hooks of one layer saw during the last forward pass of a `predict_proba`.

    Attributes:
        chunks (list): The outputs of the read module, one tensor per call: a stage may process
            the rows or the columns in chunks, each one a call of its own.
        watched (dict): For every module of `Readout.watch`, the shape of its first output in
            that pass.
        end_shape (tuple): The output shape of the `Readout.end` module, None without one.
        end_kwargs (dict): The keyword arguments of that call, None without one.
        n_test (int): The number of test rows.
    """

    chunks: list
    watched: dict
    end_shape: Optional[tuple]
    end_kwargs: Optional[dict]
    n_test: int


@dataclass
class Readout:
    """How one layer is read: which module is hooked and how its outputs become embeddings.

    Attributes:
        module (str): The dotted name of the submodule whose output is captured.
        reduce (callable): `Capture -> torch.Tensor`, the activations shaped (..., rows, dim),
            which `BaseTabularPFM._to_rows` turns into one vector per test row. Defaults to
            the output of the last call.
        end (str): A module whose output closes one forward pass ("" for the whole model), so
            that the calls of the last pass can be told apart from the ones before (one per
            ensemble batch). None keeps every call.
        watch (dict): Other modules whose output shape the reduction needs, by tag.
        pick (int): The element kept when the module returns a tuple.
        select (callable): `(output, model_kwargs) -> torch.Tensor`, applied in the hook before
            the output is copied, to keep only the part of it that is read; `model_kwargs` are
            the keyword arguments of the current call of the whole model. None keeps it all.
    """

    module: str
    reduce: Callable = lambda capture: capture.chunks[-1]
    end: Optional[str] = None
    watch: dict = field(default_factory=dict)
    pick: int = 0
    select: Optional[Callable] = None


# `last` of the models that expose their own embeddings, read by a call of their own.
NATIVE = "native"


class BaseTabularPFM(ABC):
    """Common interface of the tabular foundation models used in the experiments.

    A model wraps a scikit-learn-like classifier (fit / predict_proba) and adds
    `get_embeddings_of`, which returns the representation of the test rows at several layer
    positions, all read from one `predict_proba` call with forward hooks.

    A layer position is, in this order of lookup: a name of `_named_readouts` (the stages and
    blocks of the model, and `last`), a named depth of `LAYER_FRACTIONS`, an integer index in
    the layer stack, a short name of `LAYER_ALIASES`, or the dotted name of any submodule.

    Attributes:
        LAYER_CONTAINERS (tuple): Dotted paths tried, in order, to locate the stack of layers
            of the underlying torch model. The children of the first one found are the layers
            addressed by an integer `layer`.
        LAYER_ALIASES (dict): Short names accepted as `layer`, mapped to the dotted name of the
            submodule they stand for.
        LAYER_FRACTIONS (dict): Named depths accepted as `layer`, mapped to the fraction of the
            layer stack they stand for, for the models made of one flat stack of identical
            layers, e.g. {"early": 0.25, "middle": 0.5}. "early" reads the output of the layer
            that closes the first quarter of the stack.
        DEFAULT_MODEL_PATH (str): The checkpoint used when no `model_path` is given.
    """

    LAYER_CONTAINERS = ()
    LAYER_ALIASES = {}
    LAYER_FRACTIONS = {}
    DEFAULT_MODEL_PATH = None

    def __init__(
        self,
        model_path=None,
        n_estimators=1,
        seed=0,
        dict_params=None,
        device="cuda",
        feature_slot="target",
        verbose=True,
    ):
        """
        Args:
            model_path (str, optional): The checkpoint, None uses DEFAULT_MODEL_PATH. The
                compute nodes have no internet access, the checkpoints are downloaded by
                down_weight.py. Defaults to None.
            n_estimators (int, optional): The size of the ensemble. 1 keeps a single
                deterministic forward pass, so the captured representations are not an average
                over shuffled ensembles. Defaults to 1.
            seed (int, optional): The random state of the classifier. Defaults to 0.
            dict_params (dict, optional): Extra parameters forwarded to the classifier, to do
                ablations. Defaults to None.
            device (str, optional): The device to run the model on. Defaults to "cuda".
            feature_slot (str, optional): How per-cell representations are reduced to one
                vector per row, in [target, mean, flatten]. Only used by the layers whose
                activations are shaped (batch, rows, features, dim). Defaults to "target".
            verbose (bool, optional): Whether to print the captured shapes. Defaults to True.
        """
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.n_estimators = n_estimators
        self.seed = seed
        self.dict_params = {} if dict_params is None else dict_params
        self.device = device
        self.feature_slot = feature_slot
        self.verbose = verbose
        self._notes_shown = set()
        self.clf = self._build_classifier()

    @abstractmethod
    def _build_classifier(self):
        """Build the underlying scikit-learn-like classifier."""

    @abstractmethod
    def default_layers(self):
        """The layers `--layer all` reads, in the order of the network.

        The model has to be fitted first, since the number of blocks of each stage comes from
        the checkpoint.

        Returns:
            list: The layer names.
        """

    def _torch_model(self):
        """Return the underlying torch module of the fitted classifier."""
        model = getattr(self.clf, "model_", None) or getattr(self.clf, "model", None)
        if model is None:
            raise RuntimeError("The classifier has to be fitted before its layers can be read")
        return model

    def _named_readouts(self):
        """The layers the model names itself (stages, blocks, `last`), name -> Readout."""
        return {}

    def _module_readout(self, name):
        """How the output of the submodule `name` is read; the models override it for the
        stages whose output needs a reduction of its own."""
        return Readout(name, reduce=lambda capture: self._cells(capture.chunks[-1]))

    def _native_embeddings(self, X_test):
        """The embeddings the model exposes itself, for a `NATIVE` layer."""
        raise NotImplementedError(f"{type(self).__name__} has no native embeddings")

    def fit(self, X_train, y_train):
        """Fit the classifier on a training set.

        Args:
            X_train (np.ndarray): The train features.
            y_train (np.ndarray): The train labels.

        Returns:
            BaseTabularPFM: self.
        """
        self.clf.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def predict_proba(self, X_test):
        return self.clf.predict_proba(X_test)

    def get_embeddings(self, X_test, layer="last"):
        """Representations of the test rows at one layer position, see `get_embeddings_of`."""
        embeddings, errors = self.get_embeddings_of(X_test, [layer])
        if errors:
            raise errors[layer]
        return embeddings[layer]

    def get_embeddings_of(self, X_test, layers):
        """The embeddings at several layers, all read from one forward pass.

        Reading a layer with a hook costs a whole forward pass, so the hooks of every layer
        are put in place before a single `predict_proba` call.

        Args:
            X_test (np.ndarray): The test features.
            layers (list): The layer positions.

        Returns:
            tuple: `{layer: embeddings}`, each of shape (n_test, dim), of the layers that could
                be read, and `{layer: exception}` of those that could not.
        """
        embeddings, errors, readouts = {}, {}, {}
        for layer in layers:
            try:
                readout = self._readout(layer)
                if readout == NATIVE:
                    embeddings[layer] = self._native_embeddings(X_test)
                else:
                    readouts[layer] = readout
            except Exception as error:  # noqa: BLE001 - reported per layer
                errors[layer] = error
        if not readouts:
            return embeddings, errors

        model = self._torch_model()
        events = {layer: [] for layer in readouts}
        call = {"kwargs": None}  # the keyword arguments of the current call of the model
        handles = [
            model.register_forward_pre_hook(lambda _m, _args, kwargs: call.update(kwargs=kwargs), with_kwargs=True)
        ]
        try:
            for layer, readout in readouts.items():
                handles += _register_hooks(model, readout, events[layer], call)
            self.clf.predict_proba(X_test)
        except Exception as error:  # noqa: BLE001 - every hooked layer fails with it
            return embeddings, {**errors, **dict.fromkeys(readouts, error)}
        finally:
            for handle in handles:
                handle.remove()

        for layer, readout in readouts.items():
            try:
                capture = _last_pass(events[layer], readout, len(X_test))
                activations = readout.reduce(capture)
                if self.verbose:
                    print(f"  captured {layer} ({readout.module}): {tuple(activations.shape)}")
                embeddings[layer] = self._to_rows(activations, len(X_test))
            except Exception as error:  # noqa: BLE001 - reported per layer
                errors[layer] = error
        self.verbose = False  # only report the captures of the first pass of a run
        return embeddings, errors

    def list_layers(self):
        """Names of the submodules the embeddings can be read at.

        The model has to be fitted first, since the torch module is only built then.

        Returns:
            dict: The default layers, the (index, name) pairs of the layer stack, the named
                depths and all submodule names.
        """
        model = self._torch_model()
        try:
            stack = self._layer_stack(model)
        except AttributeError:
            stack = []  # the submodule names below are still enough to pick a layer
        return {
            "default": self.default_layers(),
            "layer_stack": [(i, name) for i, (name, _) in enumerate(stack)],
            "depths": {
                depth: stack[self._depth_index(len(stack), fraction)][0]
                for depth, fraction in self.LAYER_FRACTIONS.items()
            }
            if stack
            else {},
            "all_modules": [name for name, _ in model.named_modules() if name],
        }

    def stack_layers(self):
        """The indices of the layer stack, as layer names: "0", "1", ..."""
        return [str(i) for i in range(len(self._layer_stack(self._torch_model())))]

    @staticmethod
    def _depth_index(n_layers, fraction):
        """Index of the layer closing the first `fraction` of a stack of `n_layers`."""
        return min(n_layers - 1, max(0, int(n_layers * fraction) - 1))

    def _layer_stack(self, model):
        """Locate the stack of layers, i.e. the children of the first container found."""
        for path in self.LAYER_CONTAINERS:
            container = _resolve_path(model, path)
            if container is not None:
                return [(f"{path}.{name}", module) for name, module in container.named_children()]
        raise AttributeError(
            f"None of the layer containers {self.LAYER_CONTAINERS} were found on "
            f"{type(model).__name__}; run with --list_layers to see the available submodules"
        )

    def _readout(self, layer):
        """Turn a layer spec into the Readout of the module it addresses, or NATIVE."""
        layer = str(layer)
        named = self._named_readouts()
        if layer in named:
            return named[layer]
        model = self._torch_model()
        if layer in self.LAYER_FRACTIONS:
            stack = self._layer_stack(model)
            return self._module_readout(stack[self._depth_index(len(stack), self.LAYER_FRACTIONS[layer])][0])
        if layer.lstrip("-").isdigit():
            stack = self._layer_stack(model)
            index = int(layer)
            if not -len(stack) <= index < len(stack):
                raise IndexError(f"Layer {index} out of range, the stack has {len(stack)} layers")
            return self._module_readout(stack[index][0])
        name = self.LAYER_ALIASES.get(layer, layer)
        if _resolve_path(model, name) is None:
            raise AttributeError(f"Layer {layer} not found; run with --list_layers to see the available ones")
        return self._module_readout(name)

    def _note(self, message):
        """Print a notice once per run."""
        if message not in self._notes_shown:
            print(f"  {message}")
            self._notes_shown.add(message)

    def _cells(self, activations):
        """Reduce the cell axis of (batch, rows, cells, dim) activations by `feature_slot`, the
        last cell being the target; other shapes are returned unchanged."""
        if activations.ndim != 4:
            return activations
        if self.feature_slot == "target":
            return activations[..., -1, :]
        if self.feature_slot == "mean":
            return activations.mean(dim=-2)
        if self.feature_slot == "flatten":
            return activations.flatten(start_dim=-2)
        raise ValueError(f"Unknown feature_slot {self.feature_slot}")

    def _cells_without_target(self, activations):
        """Reduce the feature axis of (batch, rows, features, dim) activations that have no
        target cell: "target" has no cell to pick, it is read as "mean"."""
        if self.feature_slot == "flatten":
            return activations.flatten(start_dim=-2)
        if self.feature_slot == "target":
            self._note("this stage has no target cell, feature_slot 'target' is read as 'mean'")
        return activations.mean(dim=-2)

    @staticmethod
    def _to_rows(activations, n_test):
        """Reduce (..., rows, dim) activations to one embedding vector per test row.

        Args:
            activations (torch.Tensor): The reduced activations.
            n_test (int): The number of test rows.

        Returns:
            np.ndarray: The embeddings, of shape (n_test, dim), averaged over the leading axes
                (the tables of an ensemble).
        """
        activations = activations.reshape(-1, activations.shape[-2], activations.shape[-1])
        # The sequence axis holds the train rows followed by the test rows.
        return activations[:, -n_test:, :].mean(dim=0).numpy()


def _register_hooks(model, readout, events, call):
    """Put the hooks of one layer in place, appending to `events` as they fire.

    ("out", tensor) per call of the read module, (tag, shape) per call of a watched module,
    ("end", (shape, kwargs)) per call of the end module.
    """
    import torch

    def on_output(_module, _inputs, output):
        output = output[readout.pick] if isinstance(output, tuple) else output
        if readout.select is not None:
            output = readout.select(output, call["kwargs"])
        events.append(("out", output.detach().to("cpu", torch.float32, copy=True)))

    def on_watched(tag):
        return lambda _module, _inputs, output: events.append((tag, _shape(output)))

    def on_end(_module, _args, kwargs, output):
        events.append(("end", (_shape(output), kwargs)))

    handles = [model.get_submodule(readout.module).register_forward_hook(on_output)]
    for tag, name in readout.watch.items():
        handles.append(model.get_submodule(name).register_forward_hook(on_watched(tag)))
    if readout.end is not None:
        handles.append(model.get_submodule(readout.end).register_forward_hook(on_end, with_kwargs=True))
    return handles


def _last_pass(events, readout, n_test):
    """The Capture of the last forward pass in the events of one layer."""
    end_shape = end_kwargs = None
    if readout.end is None:
        window = events
    else:
        ends = [i for i, (tag, _) in enumerate(events) if tag == "end"]
        if not ends:
            raise RuntimeError(f"The hook on {readout.end or 'the model'} never fired during predict_proba")
        start = ends[-2] + 1 if len(ends) > 1 else 0
        window = events[start : ends[-1]]
        end_shape, end_kwargs = events[ends[-1]][1]
    chunks = [value for tag, value in window if tag == "out"]
    if not chunks:
        raise RuntimeError(f"The hook on {readout.module} never fired during the last forward pass")
    watched = {}
    for tag, value in window:
        if tag in readout.watch:
            watched.setdefault(tag, value)
    return Capture(chunks, watched, end_shape, end_kwargs, n_test)


def _shape(output):
    output = output[0] if isinstance(output, tuple) else output
    return tuple(output.shape) if hasattr(output, "shape") else None


def _resolve_path(module, path):
    """Follow a dotted attribute/submodule path, returning None if it does not exist."""
    current = module
    for attribute in path.split("."):
        if hasattr(current, attribute):
            current = getattr(current, attribute)
        elif attribute.isdigit() and hasattr(current, "__getitem__"):
            try:
                current = current[int(attribute)]
            except (IndexError, KeyError, TypeError):
                return None
        else:
            return None
    return current
