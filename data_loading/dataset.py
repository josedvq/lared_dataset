import pickle
import random
import logging

logger = logging.getLogger(__name__)
handler = logging.FileHandler('extractor.log')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pytorch_lightning.utilities.seed import isolate_rng

class FatherDatasetSubset(torch.utils.data.Subset):
    def __init__(self, *args, eval=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval = eval

    def __getitem__(self, idx):
        if type(idx) == list:
            return self.dataset.get_multiple_items([self.indices[i] for i in idx])
        else:
            return self.dataset.get_item(self.indices[idx])

    def auc(self, idxs, proba) -> float:
        return self.dataset.auc(idxs, proba)

    def accuracy(self, idxs, proba) -> float:
        return self.dataset.accuracy(idxs, proba)

class FatherDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        extractors:dict,
        transform=None
    ) -> None:
        self.examples = examples
        self.extractors = extractors
        self.label_threshold = 0.25
        self.transform = transform

    def get_multiple_items(self, idxs):
        examples = [self.examples[idx] for idx in idxs] 
        keys = [(ex['pid'], ex['ini_time'], ex['end_time']) for ex in examples]

        items = {}
        for ex_name, extractor in self.extractors.items():
            items[ex_name] = extractor.extract_multiple(keys)

        # items['label'] = [np.mean(ex['vad']) >= self.label_threshold for ex in examples]
        items['poses'] = np.stack([ex['poses'].transpose() for ex in examples])
        items['label'] = np.stack([ex['interp_vad'] for ex in examples])
        items['index'] = idxs

        if self.transform:
            items = self.transform(items, multiple=True)

        return items

    def get_item(self, idx, eval_mode=False) -> dict:
        item = {}
        ex = self.examples[idx]
        key = (ex['pid'], ex['ini_time'], ex['end_time'])

        item = {}
        for ex_name, extractor in self.extractors.items():
            item[ex_name] = extractor(*key)

        item['poses'] = ex['poses']
        # item['label'] = np.mean(ex['vad']) >= self.label_threshold
        item['label'] = ex['interp_vad']
        item['index'] = idx
        
        if self.transform:
            item = self.transform(item)
        
        return item

    def __getitem__(self, idx) -> dict:
        if type(idx) == list:
            return self.get_multiple_items(idx)
        else:
            return self.get_item(idx)

    def __len__(self):
        return len(self.examples)

    def get_all_labels(self):
        return [np.mean(ex['vad']) >= self.label_threshold for ex in self.examples]

    def get_groups(self):
        return [f'{ex["pid"]}' for ex in self.examples]

    def auc(self, idxs, proba: np.array):
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        return roc_auc_score(labels, proba)

    def accuracy(self, idxs, proba: np.array):
        labels = self.get_all_labels()
        labels = labels[idxs]
        assert len(labels) == len(proba)
        pred = np.argmax(proba, axis=1)

        correct = (pred == labels).sum().item()
        return correct / len(labels)


class RemoveConfidence(object):
    """ Remove the confidence from the pose matrix

    Args:
    """
    def __call__(self, items, multiple=False):
        assert 'poses' in items

        if multiple:
            s = items['poses'].shape[1]
            items['poses'] = items['poses'][:, np.mod(np.arange(s)+1, 3)!=0, :]
        else:
            raise NotImplementedError()
        return items