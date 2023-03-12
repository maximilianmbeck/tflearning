
from ml_utilities.torch_models.base_model import BaseModel
from tflearning.models.timmmodel import TimmModel
from tflearning.models.vit_timm import ViTTimm


_model_registry = {'timmmodel': TimmModel, 'vittimm': ViTTimm}


def get_model_class(model_name: str) -> BaseModel:
    """Creates the model from the model registry.

    Args:
        model_name (str): The model name.

    Returns:
        nn.Module: The model object.
    """
    model_name = model_name.lower()
    if model_name in _model_registry:
        return _model_registry[model_name]
    else:
        raise ValueError(f"Unknown model name \"{model_name}\". Available models are: {str(_model_registry.keys())}")