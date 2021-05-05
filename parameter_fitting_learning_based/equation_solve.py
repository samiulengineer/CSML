import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger
import logging
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data.dataset import random_split

logger = logging.getLogger(__name__)


"""
input_y function takes argument stack_size. 
return the 3 tensor y, x1, x2
this function creates random values from the x1 and x2
here, the eqn is y = a*x1 + b*x2
a and b are the two constants which value are 2 and 3 respectively
x1 and x2 also created randomly with the tensor size (stack_size,1)
and the shape of the y is also (stack_size)

N.B.: this function is used to create training random data
"""


def input_y(stack_size):
    x1 = torch.rand((stack_size, 1))
    x2 = torch.rand((stack_size, 1))
    a, b = 2, 3
    y = np.zeros([stack_size, 1])
    for i in range(stack_size):
        y[i] = a * x1[i] + b * x2[i]

    y = torch.Tensor(y)
    return y, x1, x2


"""
input_y_test function takes argument stack_size. 
return the 3 tensor y, x1, x2
this function creates random values from the x1 and x2. To avoid the linearity problem we multiply 2 and another random torch with same shape.
here, the eqn is y = a*x1 + b*x2
a and b are the two constants which values are 4 and 5 respectively
x1 and x2 are also created randomly with the tensor size (stack_size,1)
and the shape of the y is also (stack_size)

N.B.: this function is used to create test random data 

"""


def input_y_test(stack_size):

    x1 = torch.rand((stack_size, 1)) * torch.rand((stack_size, 1)) * 2
    x2 = torch.rand((stack_size, 1)) * torch.rand((stack_size, 1)) * 2
    a, b = 2, 3
    y = np.zeros([stack_size, 1])
    for i in range(stack_size):
        y[i] = a * x1[i] + b * x2[i]

    y = torch.Tensor(y)
    return y, x1, x2


'''
The EqnPrepare class which inherited the Dataset abstract class to create the sample data.
The len function returns the length of the dataset and the __getitem__ function returns the
supporting to fetch a data sample for a given key.
In the __init__ function we use the input_y function to create the randomm data for training.
And the __getitem__ function returns the dictionary of x1,x2 and y. 

'''


class EqnPrepare(Dataset):
    def __init__(self, stack_size=10000):

        self.stack_size = stack_size

        self.y_input, self.x1, self.x2 = input_y(self.stack_size)

    def __len__(self):
        return self.stack_size

    def __getitem__(self, idx):

        return {
            "x1": self.x1,
            "x2": self.x2,
            "y_input": self.y_input
        }


'''
The EqnTestPrepare class which inherited the Dataset abstract class to create the sample data.
The len function returns the length of the dataset and the __getitem__ function returns the
supporting to fetch a data sample for a given key.
In the __init__ function we use the input_y_test function to create the randomm data for test. here we use the default stack_size 10000.
And the __getitem__ function returns the dictionary of x1,x2 and y. 

'''


class EqnTestPrepare(Dataset):
    def __init__(self, stack_size=10000):
        self.stack_size = stack_size

        self.y_input, self.x1, self.x2 = input_y_test(self.stack_size)

    def __len__(self):
        return self.stack_size

    def __getitem__(self, idx):

        return {
            "x1": self.x1,
            "x2": self.x2,
            "y_input": self.y_input
        }


'''
The EqnDataLoader class which is inherited the LightningDataModule.
A DataModule standardizes the training, val, test splits, data preparation and transforms.
The main advantage is consistent data splits, data preparation and transforms across models.

In __init__ function we split the datset for the training and test. i.e. 80% for training and 20% for the vlidation.
 
Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset. And it tranforms the dataset in tensor.

To know more about the LightningDataModule see the documention : https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html

'''


class EqnDataLoader(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset = EqnPrepare()
        self.test_dataset = EqnTestPrepare()
        train_size = int(0.8*len(self.dataset))
        test_size = len(self.dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(
            self.dataset, [train_size, test_size])

    def train_dataloader(self):

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=4,
                                      shuffle=True,
                                      num_workers=4,
                                      )

        return train_dataloader

    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                batch_size=4,
                                shuffle=False,
                                num_workers=4)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=4,
                                 shuffle=True,
                                 num_workers=4)
        return test_loader


'''
this Eqmdel inherit the the LightningModule.
the constructor takes the argumets are input_channels, learning rate and lose_type

At first we created the model in the constructor by using nn.Linear function. 

To know about the LightningModule see the documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html

'''


class EqModel(pl.LightningModule):
    def __init__(self, in_channels=1, lr=0.05, loss_type='x_loss', *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.loss_type = loss_type
        self.in_channels = in_channels

        self.L1 = nn.Linear(self.in_channels, 64)
        self.L2 = nn.Linear(64, 32)
        self.L3 = nn.Linear(32, 16)
        self.L4 = nn.Linear(16, 8)
        self.L5 = nn.Linear(8, 4)
        self.L6 = nn.Linear(4, 2)

    # forward function returns the prediction. and also passed the data in the model
    def forward(self, y_input):
        out = self.L1(y_input)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        out = self.L5(out)
        out = self.L6(out)

        return out

    def training_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0]
        out_x2 = out[:, :, 1]

        # calculate y by using predicted x1 and x2
        recon_y = (2 * out_x1) + (3 * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        self.log
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        # self.eval()
        # with torch.no_grad():
        a, b = 2, 3
        y_input = batch['y_input']  # [B, N, 1]
        [B, N, X] = y_input.shape
        x1 = batch['x1']  # [B, N, 1]
        x2 = batch['x2']  # [B, N, 1]

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0]
        out_x2 = out[:, :, 1]

        # calculate y by using predicted x1 and x2
        recon_y = (2 * out_x1) + (3 * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)
        # self.train()
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0]
        out_x2 = out[:, :, 1]

        # calculate y by using predicted x1 and x2
        recon_y = (2 * out_x1) + (3 * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        if(self.loss_type == 'y_loss'):
            loss = F.mse_loss(recon_y, y_input)
        elif (self.loss_type == 'x_loss'):
            loss = F.mse_loss(out, ref_out)

        self.log(name='test_loss', value=loss, prog_bar=True)
        return loss


def main():

    # ------------
    # data
    # ------------

    data = EqnDataLoader()

    # ------------
    # model
    # ------------

    EqnSolve = EqModel()

    # Here the loger saved in the lightning_logs with remaned Eqn.
    # logger = TensorBoardLogger("lightning_logs", name="Eqn", log_graph=True)

    # ------------
    # training
    # ------------

    TensorBoardLogger.add_graph(logger, model=EqnSolve)
    trainer = pl.Trainer(max_epochs=100,
                         fast_dev_run=False,
                         gpus="3",
                         val_check_interval=2000,
                         logger=logger
                         )

    trainer.fit(model=EqnSolve, datamodule=data)

    # ------------
    # testing
    # ------------

    trainer.test(datamodule=data)


if __name__ == '__main__':

    main()
