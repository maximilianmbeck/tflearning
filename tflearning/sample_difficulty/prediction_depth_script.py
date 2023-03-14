import logging
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
from ml_utilities.run_utils.runner import Runner
from tflearning.sample_difficulty.prediction_depth import PredictionDepth
from tflearning.models.creator import create_model
from tflearning.data.creator import create_datasetgenerator
from ml_utilities.utils import get_device, set_seed
import wandb
LOGGER = logging.getLogger(__name__)


class PredictionDepthRunner(Runner):

    str_name = 'prediction_depth'

    def __init__(
        self,
        experiment_data: DictConfig,
        data: DictConfig,
        model: DictConfig,
        prediction_depth: DictConfig,
        **kwargs
    ):
        self.data_cfg = data
        self.model_cfg = model
        self.prediction_depth_cfg = prediction_depth
        self.experiment_data_cfg = experiment_data
        config = OmegaConf.create(
            {
                'data': data,
                'model': model,
                'prediction_depth': prediction_depth,
                'experiment_data': experiment_data
            }
        )
        # data
        ds_gen = create_datasetgenerator(self.data_cfg)
        ds_gen.generate_dataset()
        train = ds_gen.train_split
        val = ds_gen.val_split
        trainloader = DataLoader(train, batch_size=128, shuffle=False)
        if len(val) > 0:
            valloader = DataLoader(val, batch_size=128, shuffle=False)
        else:
            valloader = None

        # model
        model = create_model(model_cfg=self.model_cfg)

        # TODO: add ownjob module in creator
        # ownjob_cfg = self.model_cfg.get('ownjob', None)
        # if ownjob_cfg:
        #     job_result = JobResult(ownjob_cfg.job_dir)
        #     model = job_result.get_model_idx(ownjob_cfg.checkpoint_idx)

        # wandb init
        self._wandb_run = wandb.init(
            entity=self.experiment_data_cfg.entity,
            project=self.experiment_data_cfg.project_name,
            name=self.experiment_data_cfg.experiment_name,
            # dir=str(self.logger_directory.dir), # use default wandb dir to enable later wandb sync
            config=config,
            group=self.experiment_data_cfg.experiment_tag,
            job_type=self.experiment_data_cfg.experiment_type,
            settings=wandb.Settings(start_method='fork')
        )

        # prediction_depth
        device = get_device(self.experiment_data_cfg.gpu_id)
        prediction_depth_kwargs = OmegaConf.to_container(
            self.prediction_depth_cfg, resolve=True
        )
        prediction_depth_kwargs['train_dataloader'] = trainloader
        prediction_depth_kwargs['val_dataloader'] = valloader
        prediction_depth_kwargs['device'] = device
        prediction_depth_kwargs['experiment_specifier'] = self.experiment_data_cfg.experiment_name
        prediction_depth_kwargs['wandb_run'] = self._wandb_run
        self.prediction_depth = PredictionDepth(model, **prediction_depth_kwargs)

        set_seed(self.experiment_data_cfg.seed)

    def run(self) -> None:
        plots = self.prediction_depth.make_plots()
        # save results
        pred_depth_results = self.prediction_depth.results

        with open(Path().cwd() / 'prediction_depth_results.p', 'wb') as f:
            pickle.dump(pred_depth_results, f)

        self._wandb_run.finish()

        LOGGER.info('Done.')
