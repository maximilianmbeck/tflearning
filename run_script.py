
from pathlib import Path
from ml_utilities.utils import get_config_file_from_cli, get_config
from tflearning.scripts import run_script
from tflearning.models.vit_timm import ViTTimm

from tflearning.trainer import CovAnalysisTrainer



if __name__=='__main__':
    cfg_file = get_config_file_from_cli(config_folder='configs_scripts', script_file=Path(__file__))
    cfg = get_config(cfg_file)
    run_script(cfg)