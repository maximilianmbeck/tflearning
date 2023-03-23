from typing import Any, Dict, List, Optional
import logging
import wandb
import pickle
import torch
from dataclasses import asdict, field, dataclass
from pathlib import Path

from ml_utilities.run_utils.runner import Runner
from ml_utilities.config import ExperimentConfig
from ml_utilities.utils import get_device, set_seed

# from tflearning.models.creator import create_model, ModelConfig
from ml_utilities.torch_models import create_model, ModelConfig
from tflearning.data.creator import create_datasetgenerator, DataConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionDepthConfig:
    """Class that provides the structure for the prediction depth yaml config file.
    """
    layer_names: List[str]
    prediction_depth_mode: Optional[str] = 'last_layer_knn_prediction'
    append_softmax_output: Optional[bool] = False
    knn_n_train_samples: Optional[int] = 1000
    knn_n_neighbors: Optional[int] = 30
    batch_size: Optional[int] = 128
    knn_kwargs: Optional[Dict[str, Any]] = field(default_factory=lambda: {'n_jobs': 10})
    features_before: Optional[bool] = True
    update_bn_stats: Optional[bool] = False
    knn_preds_for_val_split: Optional[bool] = False
    use_wandb: Optional[bool] = False
    num_subsets: Optional[int] = 10


@dataclass
class PredictionDepthConfigInternal(PredictionDepthConfig):
    experiment_specifier: Optional[str] = ''
    save_dir: Optional[Path] = field(default_factory=lambda:Path('./'))
    features_before: Optional[bool] = True
    device: Optional[torch.device] = None
    wandb_run = None


@dataclass
class PredictionDepthRunnerConfig:
    """Class that provides the structure for the prediction depth runner yaml config file.
    """
    experiment_data: ExperimentConfig
    data: DataConfig
    model: ModelConfig
    prediction_depth: PredictionDepthConfig


class PredictionDepthRunner(Runner):

    str_name = 'prediction_depth'
    config_class = PredictionDepthRunnerConfig

    def __init__(self, config: PredictionDepthRunnerConfig):
        self.config = config
        set_seed(self.config.experiment_data.seed)

        # data
        ds_gen = create_datasetgenerator(self.config.data)
        ds_gen.generate_dataset()

        # model
        model = create_model(model_cfg=self.config.model)

        # wandb init
        if self.config.prediction_depth.use_wandb:
            self._wandb_run = wandb.init(
                entity=self.config.experiment_data.entity,
                project=self.config.experiment_data.project_name,
                name=self.config.experiment_data.experiment_name,
                # dir=str(self.logger_directory.dir), # use default wandb dir to enable later wandb sync
                config=asdict(config),
                group=self.config.experiment_data.experiment_tag,
                job_type=self.config.experiment_data.experiment_type,
                settings=wandb.Settings(start_method='fork'))
        else: 
            self._wandb_run = None

        # prediction_depth
        from .prediction_depth import PredictionDepth

        device = get_device(self.config.experiment_data.gpu_id)

        pd_cfg = PredictionDepthConfigInternal(**asdict(self.config.prediction_depth))
        pd_cfg.device = device
        pd_cfg.experiment_specifier = self.config.experiment_data.experiment_name
        pd_cfg.save_dir = Path().cwd()
        pd_cfg.wandb_run = self._wandb_run

        train_dataset = ds_gen.train_split
        val_dataset = None
        if self.config.prediction_depth.knn_preds_for_val_split:
            val_dataset = ds_gen.val_split

        self.prediction_depth = PredictionDepth(model=model,
                                                train_dataset=train_dataset,
                                                val_dataset=val_dataset,
                                                config=pd_cfg)

    def run(self) -> None:
        self.prediction_depth.make_plots()
        # save results
        pred_depth_results = self.prediction_depth.results

        with open(Path().cwd() / 'prediction_depth_results.p', 'wb') as f:
            pickle.dump(pred_depth_results, f)
        
        if self._wandb_run is not None:
            self._wandb_run.finish()

        LOGGER.info('Done.')
