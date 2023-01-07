import os
import sys
sys.path.append('/home/josedvq/furnace')
sys.path.append('/home/josedvq/furnace/slowfast')
import logging
import pickle
import traceback

import torch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from IPython.display import clear_output
import pytorch_lightning as pl
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
# set the cudnn
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True

from lared_dataset.data_loading.dataset import FatherDataset, FatherDatasetSubset, RemoveConfidence
from lared_dataset.data_loading.extractors import VideoExtractor, AccelExtractor
from lared_dataset.baselines.video.models import make_slow_pretrained_body as get_video_feature_extractor
from lared_dataset.baselines.video.transforms import get_kinetics_val_transform
from lared_dataset.constants import (
    processed_data_path, 
    processed_accel_path,
    processed_videos_path,
    examples_path)
from lared_dataset.baselines.train import System, train, test

def do_cross_validation(do_train, ds, input_modalities, seed, prefix=None, deterministic=False):
    
    cv_splits = list(GroupKFold(n_splits=3).split(range(len(ds)), groups=ds.get_groups()))
    
    all_results = []
    for f, (train_idx, test_idx) in enumerate(cv_splits):
        # if f == 0 or f == 1:
        #     continue
        # load feature caches for fold f
        train_ds = FatherDatasetSubset(ds, train_idx, eval=False)
        test_ds = FatherDatasetSubset(ds, test_idx, eval=True)

        weights_path = os.path.join(
            'weights',
            f'I{"-".join(input_modalities)}_fold{f}.ckpt'
        )

        pl.utilities.seed.seed_everything(seed+f+734890573)
        batch_size = 8
        if do_train:
            trainer = train(f, train_ds, test_ds, input_modalities,
                prefix = prefix + f'_fold{f}' if prefix else None,
                eval_every_epoch=True,
                deterministic=deterministic,
                weights_path=weights_path,
                batch_size=batch_size,
                num_workers=4)
            model = trainer.model
        else:
            model = System.load_from_checkpoint(checkpoint_path=weights_path)

        # ensures that the testing is reproducible regardless of training
        pl.utilities.seed.seed_everything(seed+f+2980374334)
        fold_outputs = test(f, model, test_ds, prefix = prefix + f'_fold{f}' if prefix else None, batch_size=8)
        all_results.append(fold_outputs)
        clear_output(wait=False)

    outputs = [r['proba'].numpy() for r in all_results]
    indices = [r['index'].numpy() for r in all_results]
    metrics = [r['metric'] for r in all_results]
    return metrics, outputs, indices

def do_run(examples, input_modalities, 
    do_train=True, deterministic=True, prefix=''):
    ''' train = True will train the models, and requires 
            model_label_modality = test_label_modality
        train = False will load weights to test the models and does not require
            model_label_modality = test_label_modality
    '''
    print(f'Using {len(examples)} examples')

    # create the feature datasets
    extractors = {}
    if 'video' in input_modalities:
        videos_path = processed_videos_path # os.path.join(processed_videos_path, 'examples')
        extractors['video'] = VideoExtractor(videos_path, transform=get_kinetics_val_transform(8, 256, False))

    if 'accel' in input_modalities:
        accel_ds_path = os.path.join(processed_accel_path, 'subj_accel_interp.pkl')
        extractors['accel'] = AccelExtractor(accel_ds_path)

    ds = FatherDataset(examples, extractors, transform=RemoveConfidence())

    seed = 22    
    metrics, probas, indices = do_cross_validation(
        do_train,
        ds,
        input_modalities=input_modalities,
        deterministic=deterministic,
        seed=seed,
        prefix=f'{prefix}I{"-".join(input_modalities)}')

    torch.cuda.empty_cache()

    return {
        'metrics': metrics,
        'probas': probas,
        'indices': indices,
        'seed': seed
    }

def get_table(do_train=True, deterministic=True):
    examples = pickle.load(open(examples_path, 'rb'))

    all_input_modalities = [
        # ('video',),
        # ('poses',),
        # ('accel',),
        ('video', 'accel', 'poses')
    ]    

    res = {}
    for input_modalities in all_input_modalities:

        run_results = do_run(
            examples, 
            input_modalities, 
            do_train=do_train,
            deterministic=deterministic)

        res['-'.join(input_modalities)] = run_results
    return res

try:
    res = get_table(do_train=True, deterministic=False)
except Exception:
    print(traceback.format_exc())

pickle.dump(res, open('res.pkl', 'wb'))
