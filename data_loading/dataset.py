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
            return self.dataset.get_multiple_items([self.indices[i] for i in idx], self.eval)
        else:
            return self.dataset.get_item(self.indices[idx], self.eval)

    def auc(self, idxs, proba) -> float:
        return self.dataset.auc(idxs, proba)

    def accuracy(self, idxs, proba) -> float:
        return self.dataset.accuracy(idxs, proba)

class FatherDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        extractors:dict,
        label_column = None,
        label_extractor = None,
        id_column = 'hash',
    ) -> None:
        self.examples = examples
        self.extractors = extractors
        if label_extractor is None:
            assert label_column is not None
        self.label_extractor = label_extractor
        self.label_column = label_column
        self.id_column = id_column

    def _get_start_end(self, ex, eval_mode):
        onsets = ex['onset_times'] if len(ex['onset_times']) > 0 else None
        offsets= ex['offset_times'] if len(ex['offset_times']) > 0 else None

        example_len = ex['_end_time'] - ex['_ini_time']

        if eval_mode:
            # dataset in evaluation mode
            if onsets is not None:
                # center the window around the first detection
                start, end = max(0, onsets[0]), min(example_len, offsets[0])
                mid = (end+start) / 2
            else:
                # for negative examples, select the middle section
                mid = example_len / 2
        else:
            # dataset in train mode
            if onsets is not None:
                # select a window centered within a laughter ep
                start, end = random.choice(list(zip(onsets, offsets)))
                start, end = max(0, start), min(example_len, end)
                mid = np.random.uniform(start, end)
            else:
                # for negative examples, select a section randomly
                mid = np.random.uniform(0, example_len)

        start = max(0, mid-0.75)
        end   = min(example_len, mid+0.75)

        return start, end

    def get_multiple_items(self, idxs, eval_mode=True):
        examples = [self.examples.iloc[idx,:] for idx in idxs] 
        ids = self.examples[self.id_column][idxs]
        start_end = [self._get_start_end(ex, eval_mode=eval_mode) for ex in examples]
        keys = [(id, start, end) for id, (start, end) in zip(ids, start_end)]

        items = {}
        for ex_name, extractor in self.extractors.items():
            items[ex_name] = extractor.extract_multiple(keys)

        items['intval']= np.array(start_end)
        items['index'] = idxs
        
        if self.label_extractor is not None:
            instance_ids = self.examples['instance_id'][idxs]
            conditions = self.examples['condition'][idxs]
            keys = [((instance_id, id, condition), start, end) 
                for instance_id, id, condition, (start, end) 
                in zip(instance_ids, ids, conditions, start_end)]
            items['label'] = self.label_extractor.extract_multiple(keys)
        else:
            items['label'] = self.examples[self.label_column][idxs].to_numpy()

        return items

    def get_item(self, idx, eval_mode=False) -> dict:
        item = {}
        ex = self.examples.iloc[idx,:]
        start, end = self._get_start_end(ex, eval_mode=eval_mode)

        for ex_name, extractor in self.extractors.items():
            item[ex_name] = extractor(ex[self.id_column], start=start, end=end)

        if self.label_extractor is not None:
            instance_id = self.examples['instance_id'][idx]
            id = self.examples[self.id_column][idx]
            condition = self.examples['condition'][idx]
            key = (instance_id, id, condition) 
            item['label'] = self.label_extractor(key, start, end)
        else:
            item['label'] = ex[self.label_column]

        item['intval']= [start, end]
        item['index'] = idx
        return item

    def __getitem__(self, idx) -> dict:
        if type(idx) == list:
            return self.get_multiple_items(idx)
        else:
            return self.get_item(idx)

    def __len__(self):
        return len(self.examples)

    def get_all_labels(self):
        return self.examples[self.label_column].to_numpy()

    def get_groups(self):
        return self.examples[self.id_column].to_numpy()

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