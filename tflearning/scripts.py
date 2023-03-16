from dataclasses import dataclass, field
import logging
from typing import Dict, Type, Union, Any
from omegaconf import OmegaConf, DictConfig
from dacite import from_dict

from ml_utilities.run_utils.runner import Runner

from .mode_connectivity.instability_analysis import InstabilityAnalyzer
from .mode_connectivity.train_instability_analysis import TrainInstabilityAnalysis
from .sample_difficulty.prediction_depth_script import PredictionDepthRunner

LOGGER = logging.getLogger(__name__)

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


@dataclass
class ScriptConfig:
    run_script_name: str
    kwargs: Union[DictConfig, Dict[str, Any]] = field(default_factory=dict)


def run_script(cfg: DictConfig):
    LOGGER.info(f'Running script with config: \n{OmegaConf.to_yaml(cfg)}')
    cfg = from_dict(data_class=ScriptConfig, data=cfg.config)
    script_runner_class = get_runner_script(cfg.run_script_name)
    config_class = getattr(script_runner_class, 'config_class', None)
    if config_class is not None:
        cfg.kwargs = from_dict(data_class=config_class, data=OmegaConf.to_container(cfg.kwargs, resolve=True))
        script_runner = script_runner_class(cfg.kwargs)
    else:
        script_runner = script_runner_class(**cfg.kwargs)
    script_runner.run()
