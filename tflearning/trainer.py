import torch
import logging
from torch import nn
from torch.utils import data
from typing import Dict, Any
from omegaconf import DictConfig

from ml_utilities.data.classificationdataset import ClassificationDatasetWrapper
from ml_utilities.torch_utils import gradients_to_vector
from ml_utilities.trainer.supervisedbasetrainer import SupervisedBaseTrainer
from ml_utilities.trainer import register_trainer
from ml_utilities.utils import flatten_hierarchical_dict
from .learning_dynamics.covariance_analysis import gradient_covariance_analysis

class CovAnalysisTrainer(SupervisedBaseTrainer):

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._log_additional_every = config.trainer.get('log_additional_train_step_every', 100)
        self._enable_cov_analysis = config.trainer.get('enable_cov_analysis', True)
        self._cov_analysis_args = config.trainer.get('cov_analysis_args', {})

    def _create_datasets(self) -> None:
        super()._create_datasets()
        data_cfg = self.config.data
        # create subclasses datasets if necessary
        use_classes = data_cfg.get('use_classes', None)
        if use_classes is not None:
            from ml_utilities.data import get_dataset_label_names
            label_names = get_dataset_label_names(data_cfg.dataset)
            subclasses_datasets = {
                key: ClassificationDatasetWrapper(dataset=split,
                                                  label_names=label_names).create_subclassification_dataset(
                                                      list(use_classes)) for key, split in self._datasets.items()
            }
            self._datasets = subclasses_datasets

    def _create_dataloaders(self) -> None:
        super()._create_dataloaders()
        # create cov analysis dataloaders, such that they do not interfere with the normal dataloaders
        if self._enable_cov_analysis:
            train_loader = data.DataLoader(dataset=self._datasets['train'],
                                           batch_size=self.config.trainer.batch_size,
                                           shuffle=True,
                                           drop_last=False,
                                           num_workers=self.config.trainer.num_workers,
                                           persistent_workers=True,
                                           pin_memory=True)
            val_loader = data.DataLoader(dataset=self._datasets['val'],
                                         batch_size=self.config.trainer.batch_size,
                                         shuffle=True,
                                         drop_last=False,
                                         num_workers=self.config.trainer.num_workers,
                                         persistent_workers=True,
                                         pin_memory=True)
            self._cov_analysis_loaders = {'train': train_loader, 'val': val_loader}

    def _get_additional_train_step_log(
            self, step: int) -> Dict[str, Any]:
        if step % self._log_additional_every != 0:
            return {}
        log_dict = {}
        # norm of model parameter vector
        model_param_vec = nn.utils.parameters_to_vector(self._model.parameters())
        model_param_norm = torch.linalg.norm(model_param_vec, ord=2).item()
        grad_norm_vec = gradients_to_vector(self._model.parameters())
        grad_norm = torch.linalg.norm(grad_norm_vec, ord=2).item()
        log_dict.update({'weight_norm': model_param_norm, 'grad_norm': grad_norm})

        # gradient covariance analysis
        if self._enable_cov_analysis:
            grad_cov_analysis_ret = gradient_covariance_analysis(model=self._model,
                                                                 loss_fn=self._loss,
                                                                 dataloaders=self._cov_analysis_loaders,
                                                                 device=self.device,
                                                                 **self._cov_analysis_args)
            grad_cov_analysis_dict = flatten_hierarchical_dict(grad_cov_analysis_ret[0])
            log_dict.update(grad_cov_analysis_dict)
        return log_dict

# register trainer
register_trainer('supervised_cov_analysis', CovAnalysisTrainer)