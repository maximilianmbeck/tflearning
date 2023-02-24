from omegaconf import DictConfig
import hydra
from tflearning.scripts import run_script
from tflearning.models.vit_timm import ViTTimm
from tflearning.trainer import CovAnalysisTrainer
import logging
from pathlib import Path

LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='configs_scripts_hydra', config_name='PD-config')
def main(cfg: DictConfig):
    LOGGER.info(f'Run folder: {Path.cwd().resolve()}')
    run_script(cfg)

if __name__=='__main__':
    main()