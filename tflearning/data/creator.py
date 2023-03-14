from dataclasses import dataclass
from typing import Optional, Union
from ml_utilities.config import NameAndKwargs
from omegaconf import DictConfig
from tflearning.data.sample_selectors import get_sample_selector_class


@dataclass
class DatasetGeneratorConfig:
    data_root_path: str
    n_px: int


@dataclass
class DataConfig:
    name: str
    kwargs: DatasetGeneratorConfig
    sample_selector: Optional[NameAndKwargs] = None


def create_datasetgenerator(data_cfg: Union[DictConfig, DataConfig]):
    from . import get_datasetgenerator_class

    if isinstance(data_cfg, DictConfig):
        data_cfg = DataConfig(**data_cfg)

    datasetgenerator_class = get_datasetgenerator_class(data_cfg.name)
    sample_selector = None
    if data_cfg.sample_selector is not None:
        sample_selector_class = get_sample_selector_class(data_cfg.sample_selector.name)
        sample_selector = sample_selector_class(**data_cfg.sample_selector.kwargs)

    datasetgenerator = datasetgenerator_class(
        **dict(data_cfg.kwargs), train_sample_selector=sample_selector
    )

    return datasetgenerator
