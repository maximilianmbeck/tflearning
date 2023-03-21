# adapted from https://github.com/ptrblck/pytorch_misc/blob/master/shared_dict.py
from multiprocessing import Manager
from torch.utils.data import Dataset

class CachedDataset(Dataset):

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self._manager = Manager()
        self.shared_dict = self._manager.dict()

    def __getitem__(self, index):
        if index not in self.shared_dict:
            self.shared_dict[index] = self.dataset[index]
        return self.shared_dict[index]
    
    def __getattr__(self, item):
        if item == "dataset":
            return getattr(super(), item)
        return getattr(self.dataset, item)

    def __len__(self):
        return len(self.dataset)
