import os
import sys
sys.path.append('/home/josedvq/furnace')
sys.path.append('/home/josedvq/furnace/slowfast')
import logging
import pickle
import traceback

import torch
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
import pytorch_lightning as pl
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

from lared_dataset.constants import (examples_path)
from lared_dataset.baselines.train import do_run


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
            deterministic=deterministic,
            enable_progress_bar=False)

        res['-'.join(input_modalities)] = run_results
    return res

try:
    res = get_table(do_train=True, deterministic=False)
except Exception:
    print(traceback.format_exc())

pickle.dump(res, open('res.pkl', 'wb'))
