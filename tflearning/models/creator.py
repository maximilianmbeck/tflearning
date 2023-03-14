from ml_utilities.torch_models.base_model import BaseModel
from ml_utilities.config import ModelConfig
from tflearning.models import get_model_class


def create_model(model_cfg: ModelConfig) -> BaseModel:
    model_class = get_model_class(model_cfg.name)

    model = model_class(model_cfg.kwargs)

    return model
