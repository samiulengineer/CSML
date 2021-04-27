import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torch.nn.functional as F
import pandas as pd

from pytorch_lightning.loggers import TensorBoardLogger
import logging

logger = logging.getLogger(__name__)


def input_y(stack_size):
    x1 = torch.rand((stack_size, 1))
    x2 = torch.rand((stack_size, 1))
    a, b = 2, 3
    y = np.zeros([stack_size, 1])
    for i in range(stack_size):
        y[i] = a * x1[i] + b * x2[i]

    y = torch.Tensor(y)
    return y, x1, x2


class EqnPrepare(Dataset):
    def __init__(self, stack_size=100):
        self.stack_size = stack_size
        # self.x1 = torch.rand((self.stack_size, 1))
        # self.x2 = torch.rand((self.stack_size, 1))

        self.y_input, self.x1, self.x2 = input_y(self.stack_size)

    def __len__(self):
        return self.stack_size

    def __getitem__(self, idx):
        return {
            "x1": self.x1,
            "x2": self.x2,
            "y_input": self.y_input
        }


class EqModel(pl.LightningModule):
    def __init__(self, in_channels=1, lr=1e-3, loss_type='mse', *args, **kwargs):
        super().__init__()
        self.lr = lr
        self.in_channels = in_channels
        self.loss_type = loss_type

        self.L1 = nn.Linear(1, 10)
        self.L2 = nn.Linear(10, 50)
        self.L3 = nn.Linear(50, 25)
        self.L4 = nn.Linear(25, 2)

        # call ultimate weigth init
        # self.apply(weight_init)

    def forward(self, y_input):
        # This is a quick demo case that only use phase information as input
        out = self.L1(y_input)
        out = self.L2(out)
        out = self.L3(out)
        out = self.L4(out)
        return out

    def training_step(self, batch, batch_idx):
        y_input = batch["y_input"]
        x1 = batch["x1"]
        x2 = batch["x2"]
        [B, N, X] = y_input.shape

        # loss function for out and ref_out
        ref_out = torch.cat([x1, x2], 2)
        out = self.forward(y_input)
        loss = F.mse_loss(out, ref_out)  # simple RMSE loss

        # loss function for y_input and recon_y
        out_x1 = out[:, :, 0]
        out_x2 = out[:, :, 1]
        recon_y = (2 * out_x1) + (3 * out_x2)
        recon_y = torch.reshape(recon_y, [B, N, X])

        # loss = F.mse_loss(recon_y, y_input)
        self.log('step_loss', loss, prog_bar=True,
                 logger=True, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), self.lr)
        return optimizer

    def validation_step(self, batch, batch_idx):
        self.eval()
        with torch.no_grad():
            a, b = 2, 3
            y_input = batch['y_input']  # [B, N, 1]
            x1 = batch['x1'].cpu().data.numpy()  # [B, N, 1]
            x2 = batch['x2'].cpu().data.numpy()  # [B, N, 1]
            y_pred = self.forward(y_input)  # [B, N, 2]
            out_x1 = y_pred[:, :, 0]
            out_x2 = y_pred[:, :, 1]
            recon_y = (a * out_x1) + (b * out_x2)  # y = a*x1+b*x2
            d = {'x1': x1.squeeze().tolist(), 'out_x1': out_x1.cpu().data.numpy().squeeze().tolist(
            ), 'x2': x2.squeeze().tolist(), 'out_x2':  out_x2.cpu().data.numpy().squeeze().tolist()}
            df = pd.DataFrame(data=d)
            print(df)
        self.train()

    def train_dataloader(self):

        self.train_dataset = EqnPrepare()

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=4,
                                      shuffle=True,
                                      num_workers=4,
                                      )

        return train_dataloader

    def val_dataloader(self):
        self.val_dataset = EqnPrepare()
        val_loader = DataLoader(self.val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=4,
                                drop_last=True,
                                pin_memory=True)
        return val_loader


def main():

    # ------------
    # model
    # ------------

    model = EqModel()

    # checkpoint_callback = ModelCheckpoint(filepath   = './parameter_fitting/',
    #                                       save_top_k = 1,
    #                                       monitor    = 'loss',
    #                                       verbose    = True)

    # logger = TensorBoardLogger("lightning_logs",
    #                            name="DnCNN"
    #                            )

    # ------------
    # training
    # ------------

    trainer = pl.Trainer(max_epochs=10,
                         fast_dev_run=False,
                         gpus=1,
                         val_check_interval=25
                         )

    trainer.fit(model)

    # pretrained_model = DnCNN.load_from_checkpoint("./parameter_fitting/DnCNN.ckpt")

    # ------------
    # testing
    # ------------

    # test_set = trainer.test(model)


if __name__ == '__main__':

    main()

# if __name__ == "__main__":

#     train_dataset = EqnPrepare()

#     train_loader = DataLoader(train_dataset,
#                               batch_size=4,
#                               shuffle=True,
#                               num_workers=4,
#                               drop_last=True,
#                               pin_memory=True)

#     for batch_idx, batch in enumerate(train_loader):
#         print('x1 shape \t = {}'.format(batch['x1'].shape))
#         print('x2 shape \t = {}'.format(batch['x2'].shape))
#         print('y_input shape \t = {}'.format(batch['y_input'].shape))
#         break

# x = np.random.randint(2, 10, (100, 1))
# y = np.random.randint(2, 10, (100, 1))
# print(x)
# print(y)

# d = {'x': x.squeeze().tolist(), 'y': y.squeeze().tolist()}
# df = pd.DataFrame(data=d)
# df
