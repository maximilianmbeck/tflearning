from dataclasses import dataclass, asdict
from typing import Optional
from ml_utilities.config import NameAndKwargs

from tflearning.data.transferdatasets import ImgClassificationDatasetGenerator
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


def create_datasetgenerator(data_cfg: DataConfig) -> ImgClassificationDatasetGenerator:
    from . import get_datasetgenerator_class

    datasetgenerator_class = get_datasetgenerator_class(data_cfg.name)
    sample_selector = None
    if data_cfg.sample_selector is not None:
        sample_selector_class = get_sample_selector_class(data_cfg.sample_selector.name)
        sample_selector = sample_selector_class(**data_cfg.sample_selector.kwargs)

    datasetgenerator = datasetgenerator_class(
        **asdict(data_cfg.kwargs), train_sample_selector=sample_selector
    )

    return datasetgenerator
