import logging
import os
import warnings
from abc import ABC, abstractmethod
from os.path import join
from typing import List, Optional, Union

import torch
from torch import nn

from ..models.auto.tokenization_auto import AutoTokenizer
from .composition import AdapterCompositionBlock, Fuse, Stack, parse_composition
from .configuration import AdapterConfig, AdapterFusionConfig, ModelAdaptersConfig, get_adapter_config_hash
from .hub_mixin import PushAdapterToHubMixin
from .loading import AdapterFusionLoader, AdapterLoader, PredictionHeadLoader, WeightsLoader
from .modeling import Adapter, GLOWCouplingBlock, NICECouplingBlock
from .utils import EMBEDDING_FILE, TOKENIZER_PATH, inherit_doc


logger = logging.getLogger(__name__)


class InvertibleAdaptersMixin:
    """Mixin for Transformer models adding invertible adapters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invertible_adapters = nn.ModuleDict(dict())

    def add_invertible_adapter(self, adapter_name: str):
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if adapter_name in self.invertible_adapters:
            raise ValueError(f"Model already contains an adapter module for '{adapter_name}'.")
        adapter_config = self.config.adapters.get(adapter_name)
        if adapter_config and adapter_config["inv_adapter"]:
            if adapter_config["inv_adapter"] == "nice":
                inv_adap = NICECouplingBlock(
                    [[self.config.hidden_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            elif adapter_config["inv_adapter"] == "glow":
                inv_adap = GLOWCouplingBlock(
                    [[self.config.hidden_size]],
                    non_linearity=adapter_config["non_linearity"],
                    reduction_factor=adapter_config["inv_adapter_reduction_factor"],
                )
            else:
                raise ValueError(f"Invalid invertible adapter type '{adapter_config['inv_adapter']}'.")
            self.invertible_adapters[adapter_name] = inv_adap
            self.invertible_adapters[adapter_name].apply(Adapter.init_bert_weights)

    def delete_invertible_adapter(self, adapter_name: str):
        if adapter_name in self.invertible_adapters:
            del self.invertible_adapters[adapter_name]

    def get_invertible_adapter(self):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                return self.invertible_adapters[first_adapter]
        return None

    def enable_invertible_adapters(self, adapter_names):
        for adapter_name in adapter_names:
            if adapter_name in self.invertible_adapters:
                for param in self.invertible_adapters[adapter_name].parameters():
                    param.requires_grad = True

    def invertible_adapters_forward(self, hidden_states, rev=False):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        if self.config.adapters.active_setup is not None and len(self.config.adapters.active_setup) > 0:
            first_adapter = self.config.adapters.active_setup.first()
            if first_adapter in self.invertible_adapters:
                hidden_states = self.invertible_adapters[first_adapter](hidden_states, rev=rev)

        return hidden_states


class ModelConfigAdaptersMixin(ABC):
    """
    Mixin for model config classes, adding support for adapters.

    Besides adding this mixin to the config class of a model supporting adapters, make sure the following attributes/
    properties are present: hidden_dropout_prob, attention_probs_dropout_prob.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # adapter configuration
        adapter_config_dict = kwargs.pop("adapters", None)
        if adapter_config_dict:
            self.adapters = ModelAdaptersConfig(**adapter_config_dict)
        else:
            self.adapters = ModelAdaptersConfig()
        # Convert AdapterFusions from old format for backwards compatibility
        fusion_models = kwargs.pop("adapter_fusion_models", [])
        fusion_config = kwargs.pop("adapter_fusion", None)
        for fusion_adapter_names in fusion_models:
            self.adapters.add_fusion(fusion_adapter_names, config=fusion_config)


class ModelAdaptersMixin(PushAdapterToHubMixin, ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.model_name = None
        self.loaded_embeddings = {}
        self._active_embedding = "default"

        # In some cases, the config is not an instance of a directly supported config class such as BertConfig.
        # Thus, we check the adapters config here to make sure everything is correct.
        if not hasattr(config, "adapters"):
            config.adapters = ModelAdaptersConfig()
        elif config.adapters is not None and not isinstance(config.adapters, ModelAdaptersConfig):
            config.adapters = ModelAdaptersConfig(**config.adapters)

    def _init_adapter_modules(self):
        """
        This method initializes adapter modules and fusion modules from the model config.
        """
        # Initialize adapters from config
        for adapter_name in self.config.adapters:
            self._add_adapter(adapter_name)
        # Initialize fusion from config
        for fusion_name in self.config.adapters.fusions:
            self._add_fusion_layer(fusion_name)

        self.loaded_embeddings["default"] = self.get_input_embeddings()

    # These methods have to be implemented by every deriving class:

    @abstractmethod
    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], train_embeddings=False):
        """Sets the model into mode for training the given adapters."""
        pass

    def train_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        warnings.warn(
            "add_fusion() has been deprecated in favor of add_adapter_fusion(). Please use the newer method instead.",
            FutureWarning,
        )
        self.train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)

    @abstractmethod
    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """Sets the model into mode for training of adapter fusion determined by a list of adapter names."""
        pass

    @abstractmethod
    def _add_adapter(self, adapter_name):
        pass

    @abstractmethod
    def _add_fusion_layer(self, adapter_names):
        pass

    def has_adapters(self):
        return len(self.config.adapters.adapters) > 0

    @property
    def has_parallel_adapters(self) -> bool:
        if self.config.adapters.active_setup:
            return self.config.adapters.active_setup.parallel_channels > 1
        else:
            return False

    @property
    def active_adapters(self) -> AdapterCompositionBlock:
        return self.config.adapters.active_setup

    @active_adapters.setter
    def active_adapters(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        self.set_active_adapters(adapter_setup)

    def set_active_adapters(
        self, adapter_setup: Union[list, AdapterCompositionBlock], skip_layers: Optional[List[int]] = None
    ):
        """
        Sets the adapter modules to be used by default in every forward pass. If no adapter with the given name is
        found, no module of the respective type will be activated.

        Args:
            adapter_setup (list): The list of adapters to be activated by default. Can be a fusion or stacking configuration.
        """
        adapter_setup = parse_composition(adapter_setup, model_type=self.config.model_type)
        if adapter_setup:
            for adapter_name in adapter_setup.flatten():
                if adapter_name not in self.config.adapters.adapters:
                    raise ValueError(
                        f"No adapter with name '{adapter_name}' found. Please make sure that all specified adapters are correctly loaded."
                    )

        self.config.adapters.active_setup = adapter_setup
        self.config.adapters.skip_layers = skip_layers

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False, set_active: bool = False):
        """
        Adds a new adapter module of the specified type to the model.

        Args:

            adapter_name (str): The name of the adapter module to be added.
            config (str or dict or AdapterConfig, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
            overwrite_ok (bool, optional): Overwrite an adapter with the same name if it exists. By default (False), an exception is thrown.
            set_active (bool, optional): Set the adapter to be the active one. By default (False), the adapter is added but not activated.
        """
        if isinstance(config, dict):
            config = AdapterConfig.from_dict(config)  # ensure config is ok and up-to-date
        # In case adapter already exists and we allow overwriting, explicitly delete the existing one first
        if overwrite_ok and adapter_name in self.config.adapters:
            self.delete_adapter(adapter_name)
        self.config.adapters.add(adapter_name, config=config)
        self.base_model._add_adapter(adapter_name)
        if set_active:
            self.set_active_adapters(adapter_name)

    def add_fusion(self, adapter_names: Union[Fuse, list], adapter_fusion_config=None, override_kwargs=None):
        warnings.warn(
            "add_fusion() has been deprecated in favor of add_adapter_fusion(). Please use the newer method instead.",
            FutureWarning,
        )
        adapter_fusion_config = AdapterFusionConfig.from_dict(adapter_fusion_config).replace(**override_kwargs)
        self.add_adapter_fusion(adapter_names, adapter_fusion_config)

    def add_adapter_fusion(
        self,
        adapter_names: Union[Fuse, list, str],
        config=None,
        overwrite_ok: bool = False,
        set_active: bool = False,
    ):
        """
        Adds AdapterFusion to the model with alll the necessary configurations and weight initializations

        Args:
            adapter_names (Fuse or list or str): AdapterFusion layer to add. Can be either:

                - a ``Fuse`` composition block
                - a list of adapter names to fuse
                - a comma-separated string of adapter names to fuse
            config (str or dict): adapter fusion configuration, can be either:

                - a string identifying a pre-defined adapter fusion configuration
                - a dictionary representing the adapter fusion configuration
                - the path to a file containing the adapter fusion configuration
            overwrite_ok (bool, optional): Overwrite an AdapterFusion layer with the same name if it exists. By default (False), an exception is thrown.
            set_active (bool, optional): Activate the added AdapterFusion. By default (False), the AdapterFusion is added but not activated.
        """
        if isinstance(adapter_names, Fuse):
            adapter_names = adapter_names.children
        elif isinstance(adapter_names, str):
            adapter_names = adapter_names.split(",")

        if isinstance(config, dict):
            config = AdapterFusionConfig.from_dict(config)  # ensure config is ok and up-to-date
        # In case adapter already exists and we allow overwriting, explicitly delete the existing one first
        if overwrite_ok and self.config.adapters.get_fusion(adapter_names) is not None:
            self.delete_adapter_fusion(adapter_names)
        self.config.adapters.add_fusion(adapter_names, config=config)
        self.base_model._add_fusion_layer(adapter_names)
        if set_active:
            if not isinstance(adapter_names, list):
                adapter_names = adapter_names.split(",")
            self.set_active_adapters(Fuse(*adapter_names))

    def delete_adapter(self, adapter_name: str):
        """
        Deletes the adapter with the specified name from the model.

        Args:
            adapter_name (str): The name of the adapter.
        """
        if adapter_name not in self.config.adapters:
            logger.info("No adapter '%s' found for deletion. Skipping.", adapter_name)
            return
        del self.config.adapters.adapters[adapter_name]
        self.base_model._delete_adapter(adapter_name)
        # Reset active adapters if this was the only active adapter
        if self.active_adapters == Stack(adapter_name):
            self.active_adapters = None

    def delete_adapter_fusion(self, adapter_names: Union[Fuse, list, str]):
        """
        Deletes the AdapterFusion layer of the specified adapters.

        Args:
            adapter_names (Union[Fuse, list, str]): AdapterFusion layer to delete.
        """
        if isinstance(adapter_names, Fuse):
            adapter_fusion_name = ",".join(adapter_names.children)
        elif isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        elif isinstance(adapter_names, str):
            adapter_fusion_name = adapter_names
        else:
            raise ValueError("Invalid AdapterFusion definition: {}".format(adapter_names))

        if adapter_fusion_name not in self.config.adapters.fusions:
            logger.info("No AdapterFusion '%s' found for deletion. Skipping.", adapter_fusion_name)
            return
        del self.config.adapters.fusions[adapter_fusion_name]
        self.base_model._delete_fusion_layer(adapter_fusion_name)
        # Reset active adapters if this was the active setup
        if self.active_adapters == adapter_names:
            self.active_adapters = None

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves an adapter and its configuration file to a directory so that it can be shared or reloaded using
        `load_adapter()`.

        Args:
            save_directory (str): Path to a directory where the adapter should be saved.
            adapter_name (str): Name of the adapter to be saved.

        Raises:
            ValueError: If the given adapter name is invalid.
        """
        loader = AdapterLoader(self)
        loader.save(save_directory, adapter_name, meta_dict)
        # save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_name)

    def save_adapter_fusion(
        self,
        save_directory: str,
        adapter_names: Union[Fuse, list, str],
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves an AdapterFusion layer and its configuration file to a directory so that it can be shared or reloaded
        using `load_adapter_fusion()`.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion should be saved.
            adapter_names (Union[Fuse, list, str]): AdapterFusion to be saved.

        Raises:
            ValueError: If the given AdapterFusion name is invalid.
        """
        if isinstance(adapter_names, Fuse):
            adapter_fusion_name = ",".join(adapter_names.children)
        elif isinstance(adapter_names, list):
            adapter_fusion_name = ",".join(adapter_names)
        elif isinstance(adapter_names, str):
            adapter_fusion_name = adapter_names
        else:
            raise ValueError("Invalid AdapterFusion definition: {}".format(adapter_names))

        loader = AdapterFusionLoader(self)
        loader.save(save_directory, adapter_fusion_name, meta_dict)
        # save additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.save(save_directory, adapter_fusion_name)

    def load_adapter(
        self,
        adapter_name_or_path: str,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        source: str = "ah",
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        """
        Loads a pre-trained pytorch adapter module from the local file system or a remote location.

        Args:
            adapter_name_or_path (str): can be either:

                - the identifier of a pre-trained task adapter to be loaded from Adapter Hub
                - a path to a directory containing adapter weights saved using `model.saved_adapter()`
                - a URL pointing to a zip folder containing a saved adapter module
            config (dict or str, optional): The requested configuration of the adapter.
                If not specified, will be either: - the default adapter config for the requested adapter if specified -
                the global default adapter config
            version (str, optional): The version of the adapter to be loaded.
            model_name (str, optional): The string identifier of the pre-trained model.
            load_as (str, optional): Load the adapter using this name. By default, the name with which the adapter was
                    saved will be used.
            source (str, optional): Identifier of the source(s) from where to load the adapter. Can be:

                - "ah" (default): search on AdapterHub.
                - "hf": search on HuggingFace model hub.
                - None: only search on local file system
            leave_out: Dynamically drop adapter modules in the specified Transformer layers when loading the adapter.
            set_active (bool, optional): Set the loaded adapter to be the active one. By default (False), the adapter is loaded but not activated.

        Returns:
            str: The name with which the adapter was added to the model.
        """
        loader = AdapterLoader(self)
        load_dir, load_name = loader.load(
            adapter_name_or_path,
            config,
            version,
            model_name,
            load_as,
            source=source,
            leave_out=leave_out,
            set_active=set_active,
            **kwargs,
        )
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    id2label=id2label,
                    set_active=set_active,
                )
        return load_name

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        """
        Loads a pre-trained AdapterFusion layer from the local file system.

        Args:
            adapter_fusion_name_or_path (str): a path to a directory containing AdapterFusion weights saved using `model.save_adapter_fusion()`.
            load_as (str, optional): Load the AdapterFusion using this name.
                    By default, the name with which the AdapterFusion layer was saved will be used.
            set_active (bool, optional): Activate the loaded AdapterFusion. By default (False), the AdapterFusion is loaded but not activated.

        Returns:
            str: The name with which the AdapterFusion was added to the model.
        """

        loader = AdapterFusionLoader(self)
        load_dir, load_name = loader.load(adapter_fusion_name_or_path, load_as, set_active=set_active)
        # load additional custom weights
        if custom_weights_loaders:
            for weights_loader in custom_weights_loaders:
                weights_loader.load(
                    load_dir,
                    load_as=load_as,
                    loading_info=kwargs.get("loading_info", None),
                    main_load_name=load_name,
                    set_active=set_active,
                )
        return load_name

    def save_all_adapters(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves all adapters of this model together with their configuration to subfolders of the given location.

        Args:
            save_directory (str): Path to a directory where the adapters should be saved.
        """
        for name in self.config.adapters:
            adapter_config = self.config.adapters.get(name)
            h = get_adapter_config_hash(adapter_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter(save_path, name, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders)

    def save_all_adapter_fusions(
        self,
        save_directory: str,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        """
        Saves all AdapterFusion layers of this model together with their configuration to subfolders of the given
        location.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion layers should be saved.
        """
        for name in self.config.adapters.fusions:
            adapter_fusion_config = self.config.adapters.get_fusion(name)
            h = get_adapter_config_hash(adapter_fusion_config)
            save_path = join(save_directory, name)
            if meta_dict:
                meta_dict.update({"config_id": h})
            else:
                meta_dict = {"config_id": h}
            self.save_adapter_fusion(
                save_path, name, meta_dict=meta_dict, custom_weights_loaders=custom_weights_loaders
            )

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        # first freeze/ unfreeze all model weights
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_freezed = freeze

    def pre_transformer_forward(self, **kwargs):
        """
        This method should be called by every adapter-implementing model at the very beginning of the forward() method.
        """
        # some warnings if we don't use available adapters
        active_adapters = self.active_adapters or kwargs.get("adapter_names", None)
        if not active_adapters and self.has_adapters():
            logger.warning("There are adapters available but none are activated for the forward pass.")

        self.config.adapters.is_parallelized = False

    def load_embeddings(self, path: str, name: str):
        """
        Load a saved embedding from the given path. If the embedding was saved with a tokenizer it is returned

        Args:
            path: the path to the saved embedding
            name: the name the embedding should be loaded as

        Returns: a tokenizer if it ws saved with the embedding otherwise None

        """
        if name in self.loaded_embeddings:
            raise ValueError("An embedding with the name {} already exists".format(name))
        tokenizer = None
        tokenizer_path = os.path.join(path, TOKENIZER_PATH)
        if os.path.isdir(tokenizer_path):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        embedding_path = os.path.join(path, EMBEDDING_FILE)
        if not os.path.isfile(embedding_path):
            raise FileNotFoundError("No embeddings found at {}".format(embedding_path))
        weights = torch.load(embedding_path)

        self.loaded_embeddings[name] = nn.Embedding.from_pretrained(weights)
        self.set_active_embeddings(name)
        return tokenizer

    def add_embeddings(self, name, tokenizer, reference_embedding=None, reference_tokenizer=None, embedding_dim=None):
        """
        Add a new embedding to the model. If a reference embedding and reference tokenizer are provided tokens in the
        present in both tokenizers are initialized to the embedding in the reference_embedding.

        Args:
            name: the name of the embedding
            tokenizer: the tokenizer determining the vocab of the embedding
            reference_embedding: the reference embedding to use for initializing the embeddings of tokens present in the newly created embedding
            reference_tokenizer: the tokenizer providing the vocab for the reference embedding
            embedding_dim: the dimension of the embeddings (if None the hidden_size from the config is used)

        """
        if name in self.loaded_embeddings:
            raise ValueError("An embedding with the name {} already exists".format(name))
        if embedding_dim is None:
            embedding_dim = self.config.hidden_size
        embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
        embedding.requires_grad_(False)
        if (reference_embedding is not None and reference_tokenizer is None) or (
            reference_tokenizer is not None and reference_embedding is None
        ):
            raise KeyError(
                "Reference embedding and reference tokenizer are required to use initialize embeddings from reference embedding"
            )
        if reference_embedding is not None and reference_tokenizer is not None:
            tokens = set(tokenizer.get_vocab().keys()) & set(reference_tokenizer.get_vocab().keys())
            reference_vocab = reference_tokenizer.get_vocab()
            vocab = tokenizer.get_vocab()
            for t in tokens:
                idx_reference = reference_vocab[t]
                idx = vocab[t]
                embedding.weight[idx] = self.loaded_embeddings[reference_embedding].weight[idx_reference].clone()
        embedding.train(False)
        self.loaded_embeddings[name] = embedding
        self.set_active_embeddings(name)

    def delete_embeddings(self, name):
        """
        Deletes the embedding with the given name

        Args:
            name: The name of the embedding that should be deleted

        """
        if name not in self.loaded_embeddings:
            raise ValueError("No embedding with name {}".format(name))
        if self.active_embeddings == name:
            logger.warning("The active embedding is deleted. Setting the default embedding as active.")
            self.set_active_embeddings("default")
        del self.loaded_embeddings[name]

    def save_embeddings(self, path, name, tokenizer=None):
        """
        Saves the embedding with the given name. If a tokenizer is passed as well the tokenizer is saved together with
        the embedding.

        Args:
            path: The path where the embedding should be saved
            name: The name of the embedding that should be saved
            tokenizer: optionally a tokenizer to save with the embedding (default is None)

        """
        if self.active_embeddings == name:
            self.loaded_embeddings[name] = self.get_input_embeddings()
        os.makedirs(path, exist_ok=True)
        embedding_path = os.path.join(path, EMBEDDING_FILE)
        torch.save(self.loaded_embeddings[name].weight, embedding_path)
        if tokenizer:
            tokenizer_path = os.path.join(path, TOKENIZER_PATH)
            tokenizer.save_pretrained(tokenizer_path)

    def set_active_embeddings(self, name):
        """
        Sets the active embedding for the forward pass of the model

        Args:
            name: The name of the embedding that should be used

        """
        self.loaded_embeddings[self.active_embeddings] = self.get_input_embeddings()
        self.set_input_embeddings(self.loaded_embeddings[name])
        self._active_embedding = name

    @property
    def active_embeddings(self):
        return self._active_embedding


@inherit_doc
class ModelWithHeadsAdaptersMixin(ModelAdaptersMixin):
    """
    Mixin adding support for loading/ saving adapters to transformer models with head(s).
    """

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self._convert_to_flex_head = False

    def add_adapter(self, adapter_name: str, config=None, overwrite_ok: bool = False, set_active: bool = False):
        """
        Adds a new adapter module of the specified type to the model.

        Args:
            adapter_name (str): The name of the adapter module to be added.
            config (str or dict, optional): The adapter configuration, can be either:

                - the string identifier of a pre-defined configuration dictionary
                - a configuration dictionary specifying the full config
                - if not given, the default configuration for this adapter type will be used
            overwrite_ok (bool, optional): Overwrite an adapter with the same name if it exists. By default (False), an exception is thrown.
            set_active (bool, optional): Set the adapter to be the active one. By default (False), the adapter is added but not activated.

        If self.base_model is self, must inherit from a class that implements this method, to preclude infinite
        recursion
        """
        if self.base_model is self:
            super().add_adapter(adapter_name, config, overwrite_ok=overwrite_ok, set_active=set_active)
        else:
            self.base_model.add_adapter(adapter_name, config, overwrite_ok=overwrite_ok, set_active=set_active)

    def train_adapter(self, adapter_setup: Union[list, AdapterCompositionBlock], train_embeddings=False):
        """
        Sets the model into mode for training the given adapters. If self.base_model is self, must inherit from a class
        that implements this method, to preclude infinite recursion
        """
        if self.base_model is self:
            super().train_adapter(adapter_setup, train_embeddings)
        else:
            self.base_model.train_adapter(adapter_setup, train_embeddings)

    def train_adapter_fusion(self, adapter_setup: Union[list, AdapterCompositionBlock], unfreeze_adapters=False):
        """
        Sets the model into mode for training of adapter fusion determined by a list of adapter names. If
        self.base_model is self, must inherit from a class that implements this method, to preclude infinite recursion
        """
        if self.base_model is self:
            super().train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)
        else:
            self.base_model.train_adapter_fusion(adapter_setup, unfreeze_adapters=unfreeze_adapters)

    def _add_adapter(self, adapter_name):
        """
        If self.base_model is self, must inherit from a class that implements this method, to preclude infinite
        recursion
        """
        if self.base_model is self:
            super()._add_adapter(adapter_name)
        else:
            self.base_model._add_adapter(adapter_name)

    def _add_fusion_layer(self, adapter_names):
        """
        If self.base_model is self, must inherit from a class that implements this method, to preclude infinite
        recursion
        """
        if self.base_model is self:
            super()._add_fusion_layer(adapter_names)
        else:
            self.base_model._add_fusion_layer(adapter_names)

    def save_head(self, save_directory: str, head_name: str = None):
        loader = PredictionHeadLoader(self)
        loader.save(save_directory, name=head_name)

    def load_head(self, save_directory, load_as=None, id2label=None, **kwargs):
        loader = PredictionHeadLoader(self, convert_to_flex_head=self._convert_to_flex_head)
        return loader.load(save_directory, load_as=load_as, id2label=id2label, **kwargs)

    def save_adapter(
        self,
        save_directory: str,
        adapter_name: str,
        with_head: bool = True,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().save_adapter(
            save_directory,
            adapter_name,
            meta_dict=meta_dict,
            custom_weights_loaders=custom_weights_loaders,
        )

    def load_adapter(
        self,
        adapter_name_or_path: str,
        config: Union[dict, str] = None,
        version: str = None,
        model_name: str = None,
        load_as: str = None,
        source: str = "ah",
        with_head: bool = True,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        leave_out: Optional[List[int]] = None,
        id2label=None,
        set_active: bool = False,
        **kwargs
    ) -> str:
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(
                PredictionHeadLoader(
                    self,
                    error_on_missing=False,
                    convert_to_flex_head=self._convert_to_flex_head,
                )
            )
        # Support passing a num_labels for compatibility reasons. Convert to label map here.
        num_labels = kwargs.pop("num_labels", None)
        if num_labels is not None:
            id2label = {i: "LABEL_" + str(i) for i in range(num_labels)}
        return super().load_adapter(
            adapter_name_or_path,
            config=config,
            version=version,
            model_name=model_name,
            load_as=load_as,
            source=source,
            custom_weights_loaders=custom_weights_loaders,
            leave_out=leave_out,
            id2label=id2label,
            set_active=set_active,
            **kwargs,
        )

    def save_all_adapters(
        self,
        save_directory: str,
        with_head: bool = True,
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
    ):
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().save_all_adapters(
            save_directory,
            meta_dict=meta_dict,
            custom_weights_loaders=custom_weights_loaders,
        )

    def save_adapter_fusion(
        self,
        save_directory: str,
        adapter_names: Union[Fuse, list, str],
        meta_dict: dict = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        with_head: Union[bool, str] = False,
    ):
        """
        Saves an AdapterFusion layer and its configuration file to a directory so that it can be shared or reloaded
        using `load_adapter_fusion()`.

        Args:
            save_directory (str): Path to a directory where the AdapterFusion should be saved.
            adapter_names (Union[Fuse, list, str]): AdapterFusion to be saved.
            with_head (Union[bool, str]): If True, will save a head with the same name as the AdapterFusionLayer. If a string,
                this will be used as the name of the head to be saved.

        Raises:
            ValueError: If the given AdapterFusion name is invalid.
        """
        super().save_adapter_fusion(save_directory, adapter_names, meta_dict, custom_weights_loaders)

        if with_head:
            # Make sure to cover the different options for adapter_names
            if isinstance(with_head, str):
                head_name = with_head
            elif isinstance(adapter_names, Fuse):
                head_name = adapter_names.name
            elif isinstance(adapter_names, list):
                head_name = ",".join(adapter_names)
            else:
                head_name = adapter_names
            if head_name not in self.heads:
                raise ValueError("No head with name {} found".format(head_name))
            loader = PredictionHeadLoader(self)
            loader.save(save_directory, head_name)

    def load_adapter_fusion(
        self,
        adapter_fusion_name_or_path: str,
        load_as: str = None,
        custom_weights_loaders: Optional[List[WeightsLoader]] = None,
        set_active: bool = False,
        with_head: bool = True,
        **kwargs
    ) -> str:
        if with_head:
            if custom_weights_loaders is None:
                custom_weights_loaders = []
            custom_weights_loaders.append(PredictionHeadLoader(self, error_on_missing=False))
        super().load_adapter_fusion(adapter_fusion_name_or_path, load_as, custom_weights_loaders, set_active)

    def save_all_heads(self, save_directory):
        for head_name in self.heads:
            save_path = join(save_directory, head_name)
            self.save_head(save_path, head_name)

    def get_labels(self):
        return list(self.config.id2label.values())

    def get_labels_dict(self):
        return self.config.id2label

    def get_adapter(self, name):
        """
        If self.base_model is self, must inherit from a class that implements this method, to preclude infinite
        recursion
        """
        if self.base_model is self:
            return super().get_adapter(name)
        else:
            return self.base_model.get_adapter(name)

    def load_embeddings(self, path: str, name: str):
        if self.base_model is self:
            return super().load_embeddings(path, name)
        else:
            return self.base_model.load_embeddings(path, name)

    def save_embeddings(self, path, name, tokenizer=None):
        if self.base_model is self:
            return super().save_embeddings(path, name, tokenizer)
        else:
            return self.base_model.save_embeddings(path, name, tokenizer)

    def add_embeddings(self, name, tokenizer, reference_embedding=None, reference_tokenizer=None, embedding_dim=None):
        if self.base_model is None:
            return super().add_embeddings(name, tokenizer, reference_embedding, reference_tokenizer, embedding_dim)
        else:
            return self.base_model.add_embeddings(
                name, tokenizer, reference_embedding, reference_tokenizer, embedding_dim
            )

    def set_active_embeddings(self, name):
        if self.base_model is None:
            return super().set_active_embeddings(name)
        else:
            return self.base_model.set_active_embeddings(name)

    def delete_embeddings(self, name):
        if self.base_model is None:
            return super().delete_embeddings(name)
        else:
            return self.base_model.delete_embeddings(name)
