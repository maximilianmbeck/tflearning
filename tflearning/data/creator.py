from dataclasses import dataclass, asdict
from tflearning.data.transferdatasets import ImgClassificationDatasetGenerator


@dataclass 
class DatasetGeneratorConfig:
    data_root_path: str
    n_px: int

@dataclass
class DataConfig:
    name: str
    kwargs: DatasetGeneratorConfig

def create_datasetgenerator(data_cfg: DataConfig) -> ImgClassificationDatasetGenerator:
    from . import get_datasetgenerator_class

    datasetgenerator_class = get_datasetgenerator_class(data_cfg.name)
    datasetgenerator = datasetgenerator_class(**asdict(data_cfg.kwargs))
    return datasetgenerator


    