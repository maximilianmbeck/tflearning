from torch import nn

# function that takes a model and prepares it for finetuning
# freeze some layers, takes care of output layer
# returns a new model

def freeze_layers_before(model: nn.Module, layer_name: str) -> nn.Module:
    """Freeze all layers before the given layer. The given layer is not frozen.
    Assume that the modules are ordered from input to output.
    Args:
        model (nn.Module): The model to freeze layers in.
        layer_name (str): The layer name to freeze layers before.

    Raises:
        ValueError: If the layer is not found.

    Returns:
        nn.Module: The model with frozen layers.
    """
    before = True
    
    for name, module in model.named_modules():
        if name == layer_name:
            before = False
        module.requires_grad_(not before)

    if before:
        raise ValueError('Layer {} not found'.format(layer_name))     
    return model

def adapt_last_linear_layer(model: nn.Module, num_output_logits: int) -> nn.Module:
    module_list = list(model.named_modules())
    last_layer_name, last_layer = module_list[-1]
    if not isinstance(last_layer, nn.Linear):
        raise ValueError(f'Last layer `{last_layer_name}` is not a linear layer')
    if last_layer.out_features != num_output_logits:
        # replace last layer with a new one
        new_layer = nn.Linear(last_layer.in_features, num_output_logits, last_layer.bias is not None)
        setattr(model, last_layer_name, new_layer)
    return model

def prepare_model_for_finetuning(model: nn.Module, layer_name: str = '', num_output_logits: int = -1) -> nn.Module:
    if num_output_logits > 0:
        model = adapt_last_linear_layer(model, num_output_logits)
    if layer_name:
        model = freeze_layers_before(model, layer_name)
    return model


