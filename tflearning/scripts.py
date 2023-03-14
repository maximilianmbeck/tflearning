import logging
from typing import Type
from omegaconf import OmegaConf, DictConfig

from ml_utilities.run_utils.runner import Runner

from .mode_connectivity.instability_analysis import InstabilityAnalyzer
from .mode_connectivity.train_instability_analysis import TrainInstabilityAnalysis
from .sample_difficulty.prediction_depth_script import PredictionDepthRunner

LOGGER = logging.getLogger(__name__)


KEY_RUN_SCRIPT_NAME = 'run_script_name'
KEY_RUN_SCRIPT_KWARGS = 'run_script_kwargs'


_runner_registry = {
    InstabilityAnalyzer.str_name: InstabilityAnalyzer,
    TrainInstabilityAnalysis.str_name: TrainInstabilityAnalysis,
    PredictionDepthRunner.str_name: PredictionDepthRunner
}


def get_runner_script(run_script: str) -> Type[Runner]:
    if run_script in _runner_registry:
        return _runner_registry[run_script]
    else:
        assert False, f"Unknown run script \"{run_script}\". Available run_script are: {str(_runner_registry.keys())}"


class ScriptRunner(Runner):

    def __init__(self, config: DictConfig):
        OmegaConf.resolve(config)
        self.runner_script = get_runner_script(config[KEY_RUN_SCRIPT_NAME])
        self.runner = self.runner_script(**config[KEY_RUN_SCRIPT_KWARGS])

    def run(self) -> None:
        self.runner.run()


def run_script(cfg: DictConfig):
    LOGGER.info(f'Running script with config: \n{OmegaConf.to_yaml(cfg)}')
    cfg = cfg.config
    script_runner = ScriptRunner(cfg)
    script_runner.run()
