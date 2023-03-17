from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import timm
from ml_utilities.torch_models.base_model import BaseModel
from omegaconf import DictConfig

from tflearning.finetune_utils import prepare_model_for_finetuning


@dataclass
class TimmModelConfig:
    name: str = None
    num_output_logits: Optional[int] = -1
    timm_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    freeze_layers_before: Optional[str] = ''


class TimmModel(BaseModel):
    config_class = TimmModelConfig

    def __init__(self,
                 name: str = None,
                 num_output_logits: Optional[int] = -1,
                 timm_kwargs: Optional[Dict[str, Any]] = {},
                 freeze_layers_before: Optional[str] = ''):
        super().__init__()
        self.config = TimmModelConfig(name=name,
                                      num_output_logits=num_output_logits,
                                      timm_kwargs=timm_kwargs,
                                      freeze_layers_before=freeze_layers_before)

        self.reset_parameters()

    def reset_parameters(self):
        model = timm.create_model(self.config.name, **self.config.timm_kwargs)
        self.model = prepare_model_for_finetuning(model, self.config.freeze_layers_before,
                                                  self.config.num_output_logits)

    def forward(self, x):
        return self.model(x)
