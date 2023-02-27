from omegaconf import DictConfig
import hydra
from tflearning.scripts import run_script

import logging
from pathlib import Path
import tflearning

LOGGER = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path='configs_scripts_hydra', config_name='PD-config')
def main(cfg: DictConfig):
    LOGGER.info(f'Run folder: {Path.cwd().resolve()}')
    run_script(cfg)

if __name__=='__main__':
    main()