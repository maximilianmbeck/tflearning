import logging
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Union

import torch
import torchvision.datasets as datasets
from ml_utilities.data.datasetgeneratorinterface import \
    DatasetGeneratorInterface
from PIL import Image
from torch.utils import data
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchvision.transforms import (CenterCrop, Compose, Normalize, RandomCrop,
                                    RandomHorizontalFlip, RandomResizedCrop,
                                    Resize, ToTensor)

LOGGER = logging.getLogger(__name__)

#! Transforms
def _convert_to_rgb(image):
    return image.convert('RGB')


def _default_imgs_transform(n_px: int,
                            is_train: bool,
                            normalizer_values: Dict[str, List[float]],
                            additional_train_transforms: List[Callable] = []):
    """Used for large image datasets, e.g. ImageNet, etc.
    These are the default CLIP finetuning transforms."""

    normalize = Normalize(normalizer_values['mean'], normalizer_values['std'])
    if is_train:
        return Compose([
            *additional_train_transforms,
            RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])


def _smaller_imgs_transform(n_px: int,
                            is_train: bool,
                            normalizer_values: Dict[str, List[float]],
                            additional_train_transforms: List[Callable] = []):
    """Used for small image datasets, e.g. CIFAR10, MNIST, etc."""
    normalize = Normalize(normalizer_values['mean'], normalizer_values['std'])
    if is_train:
        return Compose([
            *additional_train_transforms,
            Resize(n_px, interpolation=Image.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])


#! Metrics
def _multiclass_accuracies(num_classes: int, top_k: Union[int, List[int]] = 1) -> MetricCollection:
    if isinstance(top_k, int):
        top_k = [top_k]

    return MetricCollection(
        {f'Accuracy-top-{tk}': MulticlassAccuracy(num_classes=num_classes, top_k=tk) for tk in top_k})


class ImgClassificationDatasetGenerator(DatasetGeneratorInterface):

    def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None):
        self.data_root_path = Path(data_root_path)
        self.n_px = n_px
        self.train_sample_selector = train_sample_selector
        self._dataset_generated = False
        self.train_dataset = None
        self.val_dataset = None

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    def train_split(self) -> data.Dataset:
        return self.train_dataset

    @property
    def val_split(self) -> data.Dataset:
        return self.val_dataset

    @property
    def dataset_generated(self) -> bool:
        return self._dataset_generated


class Cifar10Generator(ImgClassificationDatasetGenerator):

    normalizer = {
        'mean': [0.4913995563983917, 0.48215848207473755, 0.44653093814849854],
        'std': [0.20230084657669067, 0.19941289722919464, 0.20096157491207123]
    }

    def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None, **kwargs):
        super().__init__(data_root_path, n_px, train_sample_selector)

    def generate_dataset(self) -> None:
        additional_train_transforms = [RandomHorizontalFlip(), RandomCrop(32, padding=4)]
        self.train_dataset = datasets.CIFAR10(self.data_root_path,
                                              train=True,
                                              download=True,
                                              transform=_smaller_imgs_transform(self.n_px, True, self.normalizer,
                                                                                additional_train_transforms))
        self.val_dataset = datasets.CIFAR10(self.data_root_path,
                                            train=False,
                                            download=True,
                                            transform=_smaller_imgs_transform(self.n_px, False, self.normalizer))

        if self.train_sample_selector is not None:
            self.train_dataset = self.train_sample_selector(self.train_dataset)

        self._dataset_generated = True

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def train_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes)

    @property
    def val_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes)


class Cifar100Generator(ImgClassificationDatasetGenerator):
    
        normalizer = {
            'mean': [0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
            'std': [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
        }
    
        def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None, **kwargs):
            super().__init__(data_root_path, n_px, train_sample_selector)
    
        def generate_dataset(self) -> None:
            additional_train_transforms = [RandomHorizontalFlip(), RandomCrop(32, padding=4)]
            self.train_dataset = datasets.CIFAR100(self.data_root_path,
                                                train=True,
                                                download=True,
                                                transform=_smaller_imgs_transform(self.n_px, True, self.normalizer,
                                                                                    additional_train_transforms))
            self.val_dataset = datasets.CIFAR100(self.data_root_path,
                                                train=False,
                                                download=True,
                                                transform=_smaller_imgs_transform(self.n_px, False, self.normalizer))
    
            if self.train_sample_selector is not None:
                self.train_dataset = self.train_sample_selector(self.train_dataset)
    
            self._dataset_generated = True
    
        @property
        def num_classes(self) -> int:
            return 100
    
        @property
        def train_metrics(self) -> MetricCollection:
            return _multiclass_accuracies(self.num_classes)
    
        @property
        def val_metrics(self) -> MetricCollection:
            return _multiclass_accuracies(self.num_classes)

class Food101Generator(ImgClassificationDatasetGenerator):

    normalizer = {
        'mean': [0.5450313091278076, 0.44354042410850525, 0.34364208579063416],
        'std': [0.2302699089050293, 0.24098420143127441, 0.2388743758201599]
    }

    def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None, **kwargs):
        super().__init__(data_root_path, n_px, train_sample_selector)

    def generate_dataset(self) -> None:
        self.train_dataset = datasets.Food101(root=self.data_root_path,
                                              split='train',
                                              download=True,
                                              transform=_default_imgs_transform(self.n_px, True, self.normalizer))
        self.val_dataset = datasets.Food101(root=self.data_root_path,
                                            split='test',
                                            download=True,
                                            transform=_default_imgs_transform(self.n_px, False, self.normalizer))

        if self.train_sample_selector is not None:
            self.train_dataset = self.train_sample_selector(self.train_dataset)

        self._dataset_generated = True

    @property
    def num_classes(self) -> int:
        return 101

    @property
    def train_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1, 5])

    @property
    def val_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1, 5])


class SUN397Generator(ImgClassificationDatasetGenerator):

    normalizer = {
        'mean': [0.47580400109291077, 0.4603254795074463, 0.4248446226119995],
        'std': [0.22625432908535004, 0.22459131479263306, 0.23804639279842377]
    }

    def __init__(self,
                 data_root_path: Union[str, Path],
                 n_px: int,
                 train_sample_selector: Callable = None,
                 train_val_split: float = 0.9,
                 seed: int = 0,
                 **kwargs):
        super().__init__(data_root_path, n_px, train_sample_selector)
        self.train_val_split = train_val_split
        self.seed = seed

    def generate_dataset(self) -> None:
        dataset = self.train_dataset = datasets.SUN397(root=self.data_root_path,
                                                       download=True,
                                                       transform=_default_imgs_transform(
                                                           self.n_px, True, self.normalizer))

        if self.train_val_split >= 1.0 or self.train_val_split <= 0.0:
            self.train_dataset = dataset
            self.val_dataset = None
        else:
            n_train_samples = int(len(dataset) * self.train_val_split)
            n_val_samples = len(dataset) - n_train_samples
            self.train_dataset, self.val_dataset = data.random_split(dataset,
                                                                     lengths=[n_train_samples, n_val_samples],
                                                                     generator=torch.Generator().manual_seed(self.seed))

        if self.train_sample_selector is not None:
            self.train_dataset = self.train_sample_selector(self.train_dataset)

        self._dataset_generated = True

    @property
    def num_classes(self) -> int:
        return 397

    @property
    def val_split(self) -> datasets.SUN397:
        if self.val_dataset is None:
            LOGGER.warning('SUN397 does not have a validation split')
        raise self.val_dataset

    @property
    def train_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1, 5])

    @property
    def val_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1, 5])


class Flowers102Generator(ImgClassificationDatasetGenerator):

    normalizer = {
        'mean': [0.4329557418823242, 0.38192424178123474, 0.2964075207710266],
        'std': [0.2588456869125366, 0.20939205586910248, 0.22115884721279144]
    }

    def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None, **kwargs):
        super().__init__(data_root_path, n_px, train_sample_selector)

    def generate_dataset(self) -> None:
        self.train_dataset = datasets.Flowers102(root=self.data_root_path,
                                                 split='train',
                                                 download=True,
                                                 transform=_default_imgs_transform(self.n_px, True, self.normalizer))
        self.val_dataset = datasets.Flowers102(root=self.data_root_path,
                                               split='val',
                                               download=True,
                                               transform=_default_imgs_transform(self.n_px, False, self.normalizer))

        if self.train_sample_selector is not None:
            self.train_dataset = self.train_sample_selector(self.train_dataset)

        self._dataset_generated = True

    @property
    def num_classes(self) -> int:
        return 102

    @property
    def train_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1, 5])

    @property
    def val_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1, 5])


class SVHNGenerator(ImgClassificationDatasetGenerator):

    normalizer = {
        'mean': [0.437682181596756, 0.4437696933746338, 0.4728044271469116],
        'std': [0.1200864240527153, 0.12313701957464218, 0.10520392656326294]
    }

    def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None, **kwargs):
        super().__init__(data_root_path, n_px, train_sample_selector)

    def generate_dataset(self) -> None:
        self.train_dataset = datasets.SVHN(root=self.data_root_path,
                                           split='train',
                                           download=True,
                                           transform=_default_imgs_transform(self.n_px, True, self.normalizer))
        self.val_dataset = datasets.SVHN(root=self.data_root_path,
                                         split='test',
                                         download=True,
                                         transform=_default_imgs_transform(self.n_px, False, self.normalizer))

        if self.train_sample_selector is not None:
            self.train_dataset = self.train_sample_selector(self.train_dataset)

        self._dataset_generated = True

    @property
    def num_classes(self) -> int:
        return 10

    @property
    def train_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1])

    @property
    def val_metrics(self) -> MetricCollection:
        return _multiclass_accuracies(self.num_classes, top_k=[1])
