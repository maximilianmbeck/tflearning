import logging
from dataclasses import dataclass

import torch
from ml_utilities.config import Config
from ml_utilities.trainer import register_trainer
from ml_utilities.trainer.universalbasetrainer import UniversalBaseTrainer
from torch import nn

from tflearning.data.creator import DataConfig, create_datasetgenerator
from tflearning.models.creator import ModelConfig, create_model

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

    def _create_dataloaders(self) -> None:
        from torch.utils import data

        # for `pin_memory` see here: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-async-data-loading-and-augmentation
        train_loader = data.DataLoader(dataset=self._datasets['train'],
                                       batch_size=self.config.trainer.batch_size,
                                       shuffle=True,
                                       drop_last=True, #! set this to true, because we have only a few samples and it is likely that the last batch is smaller than the others
                                       num_workers=self.config.trainer.num_workers,
                                       persistent_workers=True,
                                       pin_memory=True)
        val_loader = data.DataLoader(dataset=self._datasets['val'],
                                     batch_size=self.config.trainer.batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=self.config.trainer.num_workers,
                                     persistent_workers=True,
                                     pin_memory=True)
        self._loaders = dict(train=train_loader, val=val_loader)
        
        

# register trainer
register_trainer('finetune', FinetuneTrainer)