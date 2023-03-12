from typing import Type
from ml_utilities.data.datasetgeneratorinterface import DatasetGeneratorInterface
from tflearning.data.transferdatasets import Cifar10Generator

_datasetgenerator_registry = {
    'cifar10': Cifar10Generator,
}

def get_datasetgenerator_class(dataset_name: str) -> Type[DatasetGeneratorInterface]:
    if dataset_name in _datasetgenerator_registry:
        return _datasetgenerator_registry[dataset_name]
    else:
        assert False, f"Unknown dataset name \"{dataset_name}\". Available datasets are: {str(_datasetgenerator_registry.keys())}"