from dataclasses import asdict, dataclass
from dacite import from_dict
from typing import Optional, Union

from ml_utilities.config import NameAndKwargs
from ml_utilities.data import create_datasetgenerator as create_generic_datasetgenerator
from omegaconf import DictConfig

from tflearning.data.sample_selectors import get_sample_selector_class
from tflearning.data.transferdatasets import ImgClassificationDatasetGenerator


@dataclass
class DatasetGeneratorConfig:
    data_root_path: str
    n_px: int


@dataclass
class DataConfig:
    name: str
    kwargs: DatasetGeneratorConfig
    sample_selector: Optional[NameAndKwargs] = None


def create_datasetgenerator(
    data_cfg: Union[DictConfig, DataConfig]
) -> ImgClassificationDatasetGenerator:

    if data_cfg.name == 'generic':
        from ml_utilities.data.datasetgenerator import DatasetGeneratorConfig as GenericDatasetGeneratorConfig
        data_cfg = NameAndKwargs(**data_cfg)
        data_cfg.kwargs = GenericDatasetGeneratorConfig(**data_cfg.kwargs)
        return create_generic_datasetgenerator(data_cfg.kwargs)
    
    from . import get_datasetgenerator_class

    if isinstance(data_cfg, DictConfig):
        data_cfg = DataConfig(**data_cfg)

    datasetgenerator_class = get_datasetgenerator_class(data_cfg.name)
    sample_selector = None
    if data_cfg.sample_selector is not None:
        sample_selector_class = get_sample_selector_class(data_cfg.sample_selector.name)
        sample_selector = sample_selector_class(**data_cfg.sample_selector.kwargs)

    if isinstance(data_cfg.kwargs, (DictConfig, dict)):
        datasetgenerator = datasetgenerator_class(**data_cfg.kwargs, train_sample_selector=sample_selector)
    else:
        datasetgenerator = datasetgenerator_class(
            **asdict(data_cfg.kwargs), train_sample_selector=sample_selector
        )

    return datasetgenerator
