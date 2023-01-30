from typing import Union
import copy
import logging
from omegaconf import OmegaConf, DictConfig, ListConfig
from pathlib import Path
from ml_utilities.run_utils.runner import Runner, run_job, run_sweep
from ml_utilities.trainer import get_trainer_class
from .instability_analysis import InstabilityAnalyzer
"""This script automates the training for instability analysis."""

LOGGER = logging.getLogger(__name__)

IA_EXPNAME = 'IA-{stage}-{experiment_name}'


class TrainInstabilityAnalysis(Runner):

    str_name = 'train_instability_analysis'

    def __init__(self,
                 run_config: DictConfig,
                 job_config: DictConfig,
                 instability_analysis_config: Union[DictConfig, ListConfig],
                 start_num: int = 0,
                 main_training_job_dir: str = None,
                 resume_training_sweep_dir: str = None):
        self.run_config = run_config
        self.job_config = job_config
        self.start_num = start_num
        if not isinstance(instability_analysis_config, (ListConfig, list)):
            self.instability_analysis_config = [instability_analysis_config]
        else:
            self.instability_analysis_config = instability_analysis_config

        self.main_training_job_dir = main_training_job_dir
        self.resume_training_sweep_dir = resume_training_sweep_dir

        self.gpu_id = self.run_config.gpu_ids[0]
        self.save_every_idxes = list(self.instability_analysis_config[0].init_model_idxes_ks_or_every)

        self.main_seed = self.job_config.experiment_data.seed
        s = self.main_seed
        self.resume_seeds = [s + 1]
        # parameter name of the checkpoint_idx in the config
        self.init_model_idx_k_param_name = 'trainer.resume_training.checkpoint_idx'

    def create_main_training_config(self) -> DictConfig:
        # override: save_every_idxes, gpu_id, experiment_name
        main_job_cfg = copy.deepcopy(self.job_config)
        main_job_cfg.experiment_data.experiment_name = IA_EXPNAME.format(
            stage='A', experiment_name=self.job_config.experiment_data.experiment_name)
        main_job_cfg.experiment_data.gpu_id = self.gpu_id
        main_job_cfg.trainer.save_every_idxes = self.save_every_idxes

        runnable_main_job_cfg = OmegaConf.create()
        runnable_main_job_cfg.start_num = self.start_num
        runnable_main_job_cfg.config = main_job_cfg
        return runnable_main_job_cfg

    def create_resume_training_config(self, main_training_job_dir: Path) -> DictConfig:
        # override: sweep, seeds, resume_training.job_dir, resume_training.checkpoint_idx, experiment_name
        sweep_cfg = OmegaConf.create()
        sweep_cfg.type = 'line'
        sweep_cfg.axes = [{'parameter': self.init_model_idx_k_param_name, 'vals': self.save_every_idxes}]

        resume_job_cfg = copy.deepcopy(self.job_config)
        resume_job_cfg.trainer.resume_training = OmegaConf.create({
            'job_dir': str(main_training_job_dir),
            'checkpoint_idx': 'X'
        })
        resume_job_cfg.experiment_data.experiment_name = IA_EXPNAME.format(
            stage='B', experiment_name=self.job_config.experiment_data.experiment_name)

        runnable_resume_sweep_cfg = OmegaConf.create()
        runnable_resume_sweep_cfg.run_config = self.run_config
        runnable_resume_sweep_cfg.seeds = self.resume_seeds
        runnable_resume_sweep_cfg.start_num = self.start_num
        runnable_resume_sweep_cfg.config = resume_job_cfg
        runnable_resume_sweep_cfg.sweep = sweep_cfg
        return runnable_resume_sweep_cfg

    def create_instability_analysis_config(self, instability_analysis_cfg: DictConfig, main_training_job_dir: Path,
                                           resume_training_sweep_dir: Path) -> DictConfig:
        # override: instability_sweep, device, batch_size, init_model_idx_k_param_name
        runnable_ia_cfg = copy.deepcopy(instability_analysis_cfg)
        runnable_ia_cfg.main_training_job = str(main_training_job_dir)
        runnable_ia_cfg.instability_sweep = str(resume_training_sweep_dir)
        runnable_ia_cfg.device = self.gpu_id
        runnable_ia_cfg.init_model_idx_k_param_name = self.init_model_idx_k_param_name
        # use two times the training batch size since we are only evaluating
        runnable_ia_cfg.interpolate_linear_kwargs.update(OmegaConf.create(
            {'dataloader_kwargs': {
                'batch_size': 2 * self.job_config.trainer.batch_size
            }}))
        return runnable_ia_cfg

    def run(self):
        LOGGER.info('Starting TRAIN INSTABILITY ANALYSIS (IA)..')

        if self.main_training_job_dir is None and self.resume_training_sweep_dir is None:
            LOGGER.info('IA STAGE A: main training run')
            main_training_cfg = self.create_main_training_config()
            trainer_class = get_trainer_class(main_training_cfg.config.trainer.training_setup)
            self.main_training_job_dir = run_job(cfg=main_training_cfg, trainer_class=trainer_class)
        LOGGER.info(f'IA STAGE A: Done. main_training_job_dir: {self.main_training_job_dir}')

        if self.resume_training_sweep_dir is None:
            LOGGER.info('IA STAGE B: create resume runs from main training runs')
            resume_training_cfg = self.create_resume_training_config(self.main_training_job_dir)
            self.resume_training_sweep_dir = run_sweep(resume_training_cfg)
        LOGGER.info(f'IA STAGE B: Done. resume_training_sweep_dir: {self.resume_training_sweep_dir}')

        LOGGER.info('IA STAGE C: instability analysis')
        for i, ia_cfg in enumerate(self.instability_analysis_config):
            LOGGER.info(f'C-{i}: save_folder_suffix: {ia_cfg.get("save_folder_suffix", "")}')
            instability_analysis_cfg = self.create_instability_analysis_config(ia_cfg, self.main_training_job_dir,
                                                                               self.resume_training_sweep_dir)
            instability_analyzer = InstabilityAnalyzer(**instability_analysis_cfg)
            instability_analyzer.run()
        self.runner_dir = self.resume_training_sweep_dir
        LOGGER.info(f'IA STAGE C: Done. Instability analysis can be found in folder: {self.runner_dir}')
        LOGGER.info('IA Done.')
