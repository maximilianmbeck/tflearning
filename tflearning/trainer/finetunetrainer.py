import torch
import logging
from torch import nn
from torch.utils import data
from omegaconf import DictConfig

from ml_utilities.data.classificationdataset import ClassificationDatasetWrapper
from ml_utilities.trainer.supervisedbasetrainer import SupervisedBaseTrainer
from ml_utilities.trainer import register_trainer

LOGGER = logging.getLogger(__name__)

class FinetuneTrainer(SupervisedBaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._finetune_cfg = config.model.get('finetune', None)

    def _create_model(self) -> None:
        super()._create_model()
        from tflearning.finetune_utils import prepare_model_for_finetuning
        self._model = prepare_model_for_finetuning(self._model, self._finetune_cfg.layer_name, self._finetune_cfg.num_output_logits)
        
        

# register trainer
register_trainer('finetune', FinetuneTrainer)