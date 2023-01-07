import os
from functools import partial

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from .model import SegmentationFusionModel

class System(pl.LightningModule):
    def __init__(self, modalities, task='classification'):
        super().__init__()
        self.save_hyperparameters()
       
        self.model = SegmentationFusionModel(modalities, mask_len=60)

        self.loss_fn = {
            'classification':F.binary_cross_entropy_with_logits,
            'regression': F.mse_loss,
        }[task]

        self.performance_metric = {
            'classification': lambda input, target: roc_auc_score(target.flatten(), input.flatten()),
            'regression': F.mse_loss,
        }[task]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        output = self.model(batch).squeeze().cpu()
        loss = self.loss_fn(output, batch['label'].float().cpu())

        # Logging to TensorBoard by default
        self.log("train_loss", loss.detach())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        # optimizer = torch.optim.SGD(self.parameters(), lr=.001, momentum=0.9)
        return optimizer

    def validation_step(self, batch, batch_idx):
        output = self.model(batch).detach().cpu()
        
        val_loss = self.loss_fn(output.squeeze(), batch['label'].float().cpu())
        self.log('val_loss', val_loss)

        return (output, batch['label'])

    def validation_epoch_end(self, validation_step_outputs):
        all_outputs = torch.cat([o[0] for o in validation_step_outputs]).cpu()
        all_labels = torch.cat([o[1] for o in validation_step_outputs]).cpu()

        val_metric = self.performance_metric(all_outputs, all_labels)
        self.log('val_metric', val_metric)

    def test_step(self, batch, batch_idx):
        output = self.model(batch).squeeze()

        return (output, batch['index'], batch['label'])

    def test_epoch_end(self, test_step_outputs):
        all_outputs = torch.cat([o[0] for o in test_step_outputs]).cpu()
        all_indices = torch.cat([o[1] for o in test_step_outputs]).cpu()
        all_labels = torch.cat([o[2] for o in test_step_outputs]).cpu()

        test_metric = self.performance_metric(all_outputs, all_labels)
        self.test_results = {
            'metric': test_metric,
            'index': all_indices,
            'proba': all_outputs
        }
        self.log('test_metric', test_metric)

def _collate_fn(batch):
    batch = batch[0]
    return {k: torch.tensor(v) for k,v in batch.items()}

def train(i, train_ds, val_ds, modalities, 
        trainer_params={}, prefix=None, task='classification', 
        deterministic=False, eval_every_epoch=False, weights_path=None,
        batch_size=32, num_workers=8):

    num_epochs = {
        ('accel',): 10,
        ('poses',): 15,
        ('video',): 15,
        ('video', 'accel', 'poses'): 15
    }

    # data loaders
    g = torch.Generator()
    g.manual_seed(729387+i)

    data_loader_train = DataLoader(
        dataset=train_ds,
        # This line below!
        sampler=BatchSampler(
            RandomSampler(train_ds, generator=g), batch_size=batch_size, drop_last=False
        ),
        num_workers=num_workers,
        generator=g,
        collate_fn=_collate_fn
    )

    # g = torch.Generator()
    # g.manual_seed(897689769+i)
    data_loader_val = DataLoader(
        dataset=val_ds,
        # This line below!
        sampler=BatchSampler(
            SequentialSampler(val_ds), batch_size=batch_size, drop_last=False
        ),
        num_workers=num_workers,
        generator=g,
        collate_fn=_collate_fn
    )

    system = System(modalities, task=task)
    trainer_fn = partial(pl.Trainer, **trainer_params)
    trainer = trainer_fn(
        accelerator='gpu',
        check_val_every_n_epoch=1 if eval_every_epoch else 10000,
        max_epochs=num_epochs[modalities],
        logger= pl.loggers.TensorBoardLogger(
            save_dir='logs/', name='', 
            version=prefix),
        deterministic=deterministic,
        enable_checkpointing=False)
        
    trainer.fit(system, data_loader_train, data_loader_val)

    if weights_path is not None:
        trainer.save_checkpoint(weights_path)
    
    return trainer #system.test_results

def test(i, model, test_ds, prefix=None, batch_size=32):

    # data loaders
    g = torch.Generator()
    g.manual_seed(897689769+i)
    test_dl = DataLoader(
        dataset=test_ds,
        # This line below!
        sampler=BatchSampler(
            SequentialSampler(test_ds), batch_size=batch_size, drop_last=False
        ),
        num_workers=0,
        generator=g,
        collate_fn=_collate_fn
    )

    trainer = pl.Trainer(
                logger= pl.loggers.TensorBoardLogger(
                    save_dir='logs/', name='', 
                    version=prefix))

    trainer.test(
        model=model, 
        dataloaders=test_dl)
    
    return trainer.model.test_results