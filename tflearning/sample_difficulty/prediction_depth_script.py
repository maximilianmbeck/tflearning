import logging
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import timm
import pickle
from pathlib import Path
from ml_utilities.run_utils.runner import Runner
from ml_utilities.output_loader.result_loader import JobResult
from tflearning.sample_difficulty.prediction_depth import PredictionDepth
from ml_utilities.data.datasetgenerator import DatasetGenerator
from ml_utilities.utils import get_device
LOGGER = logging.getLogger(__name__)

class PredictionDepthRunner(Runner):
    
    str_name = 'prediction_depth'

    def __init__(self, data: DictConfig, model: DictConfig, prediction_depth: DictConfig, **kwargs):
        self.data_cfg = data
        self.model_cfg = model
        self.prediction_depth_cfg = prediction_depth

        ## data
        ds_gen = DatasetGenerator(**self.data_cfg)
        ds_gen.generate_dataset()
        train = ds_gen.train_split
        val = ds_gen.val_split
        trainloader = DataLoader(train, batch_size=128, shuffle=False)
        valloader = DataLoader(val, batch_size=128, shuffle=False)
        
        ## model
        timm_cfg = self.model_cfg.get('timm', None)
        if timm_cfg: 
            model = timm.create_model(**timm_cfg)
        ownjob_cfg = self.model_cfg.get('ownjob', None)
        if ownjob_cfg:
            job_result = JobResult(ownjob_cfg.job_dir)
            model = job_result.get_model_idx(ownjob_cfg.checkpoint_idx)

        ## prediction_depth
        device = get_device(self.prediction_depth_cfg.gpu_id)
        prediction_depth_kwargs = OmegaConf.to_container(self.prediction_depth_cfg, resolve=True)
        prediction_depth_kwargs['train_dataloader'] = trainloader
        prediction_depth_kwargs['val_dataloader'] = valloader
        prediction_depth_kwargs['device'] = device
        self.prediction_depth = PredictionDepth(model, **prediction_depth_kwargs)

    def run(self) -> None:
        plots = self.prediction_depth.make_plots()
        # save results
        pred_depth_results = self.prediction_depth.results

        with open(Path().cwd() / 'prediction_depth_results.p', 'wb') as f:
            pickle.dump(pred_depth_results, f)

        LOGGER.info('Done.')