import torchvision.datasets as datasets
from abc import abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Union
from ml_utilities.data.datasetgeneratorinterface import DatasetGeneratorInterface

from torchvision.transforms import Compose, Normalize, RandomResizedCrop, Resize, CenterCrop, ToTensor, RandomHorizontalFlip, RandomCrop
from PIL import Image

from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy


#! Transforms
# USE these for other datasets, (These are default transforms for CLIP finetuning)
def _convert_to_rgb(image):
    return image.convert('RGB')


# def _transform(n_px: int, is_train: bool):
#     normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#     if is_train:
#         return Compose([
#             RandomResizedCrop(n_px, scale=(0.9, 1.0), interpolation=Image.BICUBIC),
#             _convert_to_rgb,
#             ToTensor(),
#             normalize,
#         ])
#     else:
#         return Compose([
#             Resize(n_px, interpolation=Image.BICUBIC),
#             CenterCrop(n_px),
#             _convert_to_rgb,
#             ToTensor(),
#             normalize,
#         ])


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
def _multiclass_accuracy(num_classes: int, top_k: int = 1) -> MetricCollection:
    return MetricCollection({f'Accuracy-top-{top_k}': MulticlassAccuracy(num_classes=num_classes, top_k=top_k)})


class ImgClassificationDatasetGenerator(DatasetGeneratorInterface):

    def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None):
        self.data_root_path = Path(data_root_path)
        self.n_px = n_px
        self.train_sample_selector = train_sample_selector
        self._dataset_generated = False

    @property
    @abstractmethod
    def num_classes(self) -> int:
        pass

    @property
    def dataset_generated(self) -> bool:
        return self._dataset_generated


class Cifar10Generator(ImgClassificationDatasetGenerator):

    normalizer = {
        'mean': [0.4913995563983917, 0.48215848207473755, 0.44653093814849854],
        'std': [0.20230084657669067, 0.19941289722919464, 0.20096157491207123]
    }

    def __init__(self, data_root_path: Union[str, Path], n_px: int, train_sample_selector: Callable = None):
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
    def train_split(self) -> datasets.CIFAR10:
        return self.train_dataset

    @property
    def val_split(self) -> datasets.CIFAR10:
        return self.val_dataset

    @property
    def train_metrics(self) -> MetricCollection:
        return _multiclass_accuracy(self.num_classes)
    
    @property
    def val_metrics(self) -> MetricCollection:
        return _multiclass_accuracy(self.num_classes)