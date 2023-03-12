import torch
import logging
from torch import nn
from dataclasses import dataclass

from ml_utilities.trainer.universalbasetrainer import UniversalBaseTrainer
from ml_utilities.config import Config
from ml_utilities.trainer import register_trainer

from tflearning.models.creator import create_model, ModelConfig
from tflearning.data.creator import create_datasetgenerator, DataConfig

LOGGER = logging.getLogger(__name__)

@dataclass
class FinetuneConfig(Config):
    model: ModelConfig
    data: DataConfig

class FinetuneTrainer(UniversalBaseTrainer):
    config_class = FinetuneConfig

    def __init__(self, config: FinetuneConfig):
        super().__init__(config, model_init_func=create_model, datasetgenerator_init_func=create_datasetgenerator)

    def _create_model(self) -> None:
        self.config.model.kwargs['num_output_logits'] = self._datasetgenerator.num_classes
        super()._create_model()
        
        

# register trainer
register_trainer('finetune', FinetuneTrainer)