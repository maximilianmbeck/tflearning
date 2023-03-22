from typing import Type

from ml_utilities.data.datasetgeneratorinterface import \
    DatasetGeneratorInterface
from ml_utilities.data.datasetgenerator import DatasetGenerator

from tflearning.data.transferdatasets import (Cifar10Generator,
                                              Cifar100Generator,
                                              Flowers102Generator,
                                              Food101Generator,
                                              SUN397Generator, SVHNGenerator)

_datasetgenerator_registry = {
    'cifar10': Cifar10Generator,
    'cifar100': Cifar100Generator,
    'svhn': SVHNGenerator,
    'food101': Food101Generator,
    'flowers102': Flowers102Generator,
    'sun397': SUN397Generator, 
    'generic': DatasetGenerator
}


def get_datasetgenerator_class(dataset_name: str) -> Type[DatasetGeneratorInterface]:
    if dataset_name in _datasetgenerator_registry:
        return _datasetgenerator_registry[dataset_name]
    else:
        assert False, f"Unknown dataset name \"{dataset_name}\". Available datasets are: {str(_datasetgenerator_registry.keys())}"