from collections.abc import Callable
from typing import Optional, Dict

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from steering_vectors.layer_matching import (
    ModelLayerConfig,
    collect_matching_layers,
    guess_and_enhance_layer_config,
)
from steering_vectors.torch_utils import get_module

from steering_vectors.steering_vector import SteeringVector, SteeringPatchHandle, _create_additive_hook


PatchDeltaOperator = Callable[[Tensor, Tensor], Tensor]


class LayerSpecificMultipliersSteeringVector(SteeringVector):

    def patch_activations(
        self,
        model: nn.Module,
        layer_config: ModelLayerConfig | None = None,
        operator: PatchDeltaOperator | None = None,
        multiplier: Optional[Dict[int, float]] = None,
        min_token_index: int | None = None,
        token_indices: list[int] | slice | Tensor | None = None,
    ) -> SteeringPatchHandle:
        """
        Only change:
        multiplier: A multiplier for each layer to scale the patch activations. Default is 1.0.
        """
        if multiplier is None:
            multiplier = {layer_num: 1.0 for layer_num in self.layer_activations.keys()}
        else:
            assert multiplier.keys() == self.layer_activations.keys()
        assert (min_token_index is None) or (token_indices is None), (
            "Can not pass both min_token_index and token_indices"
        )
        if isinstance(token_indices, Tensor):
            assert torch.all(
                torch.logical_or(token_indices == 0, token_indices == 1)
            ), "token_indices tensor must be a mask (containing only 0s and 1s)"
        token_indices = (
            token_indices if token_indices is not None else slice(min_token_index, None)
        )
        layer_config = guess_and_enhance_layer_config(
            model, layer_config, self.layer_type
        )
        hooks: list[RemovableHandle] = []
        if self.layer_type not in layer_config:
            raise ValueError(
                f"layer_type {self.layer_type} not provided in layer config"
            )
        matcher = layer_config[self.layer_type]
        matching_layers = collect_matching_layers(model, matcher)
        for layer_num, target_activation in self.layer_activations.items():
            layer_name = matching_layers[layer_num]

            target_activation = multiplier[layer_num] * self.layer_activations[layer_num]

            module = get_module(model, layer_name)
            handle = module.register_forward_hook(
                # create the hook via function call since python only creates new scopes on functions
                _create_additive_hook(
                    target_activation.reshape(1, 1, -1), token_indices, operator
                )
            )
            hooks.append(handle)
        return SteeringPatchHandle(hooks)
