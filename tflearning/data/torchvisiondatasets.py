import logging
from pathlib import Path
from typing import Dict, List, Union
from ml_utilities.data import register_dataset
from ml_utilities.data.torchbuiltindatasets import TorchBuiltInDataset

from torchvision import datasets
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

Food101_NORMALIZER = {
    'mean': [0.5450313091278076, 0.44354042410850525, 0.34364208579063416],
    'std': [0.2302699089050293, 0.24098420143127441, 0.2388743758201599]
}
SUN397_NORMALIZER = {
    'mean': [0.47580400109291077, 0.4603254795074463, 0.4248446226119995],
    'std': [0.22625432908535004, 0.22459131479263306, 0.23804639279842377]
}
Flowers102_NORMALIZER = {
    'mean': [0.4329557418823242, 0.38192424178123474, 0.2964075207710266],
    'std': [0.2588456869125366, 0.20939205586910248, 0.22115884721279144]
}
SVHN_NORMALIZER = {
    'mean': [0.437682181596756, 0.4437696933746338, 0.4728044271469116],
    'std': [0.1200864240527153, 0.12313701957464218, 0.10520392656326294]
}


class Food101(TorchBuiltInDataset):
    def __init__(self,
                 data_root_path: Union[str, Path],
                 split: str = 'train',
                 normalizer: Union[str, Dict[str, Union[float, List[float]]]] = Food101_NORMALIZER):
        self._normalizer = normalizer

        self.dataset = datasets.Food101(root=data_root_path, split=split, download=True)
    
class SUN397(TorchBuiltInDataset):
    def __init__(self,
                 data_root_path: Union[str, Path],
                 split: str = 'train',
                 normalizer: Union[str, Dict[str, Union[float, List[float]]]] = SUN397_NORMALIZER):
        self._normalizer = normalizer
        if split != 'train':
            raise ValueError('SUN397 does not have a test set.')
        self.dataset = datasets.SUN397(root=data_root_path, download=True)

class Flowers102(TorchBuiltInDataset):
    """This dataset is resized to 224x224."""
    def __init__(self,
                 data_root_path: Union[str, Path],
                 split: str = 'train',
                 normalizer: Union[str, Dict[str, Union[float, List[float]]]] = Flowers102_NORMALIZER):
        self._normalizer = normalizer
        self.dataset = datasets.Flowers102(root=data_root_path, split=split, download=True)

class SVHN(TorchBuiltInDataset):
    """This dataset has size 32x32."""
    def __init__(self,
                 data_root_path: Union[str, Path],
                 split: str = 'train',
                 normalizer: Union[str, Dict[str, Union[float, List[float]]]] = SVHN_NORMALIZER):
        self._normalizer = normalizer
        self.dataset = datasets.SVHN(root=data_root_path, split=split, download=True)

register_dataset('Food101', Food101)
register_dataset('SUN397', SUN397)
register_dataset('Flowers102', Flowers102)
register_dataset('SVHN', SVHN)