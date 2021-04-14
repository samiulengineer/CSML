from dataloader_random import SpatialTemporalDataset
import torch.nn.functional as F
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import numpy as np
# import matplotlib.pyplot as plt


''' Hyperparameter '''

# learning_rate = 1e-3
# hidden_dim = 128
# batch_size = 32


def conv_block(input_size, output_size):
    block = nn.Sequential(
        nn.Conv2d(input_size, output_size, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_size)
    )
    return block


class DnCNN(pl.LightningModule):

    def __init__(self):
        super(DnCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=70, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = conv_block(64, 64)
        self.conv3 = conv_block(64, 64)
        self.conv4 = conv_block(64, 64)
        self.conv5 = conv_block(64, 64)
        self.conv6 = conv_block(64, 64)
        self.conv7 = conv_block(64, 64)
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=64,  kernel_size=3, padding=1)

        self.ln1 = nn.Linear(64 * 28 * 28, 128)
        # self.relu = nn.ReLU()
        # self.batchnorm = nn.BatchNorm1d(16)
        # self.dropout = nn.Dropout2d(0.5)
        self.ln2 = nn.Linear(128, 2)

        ''' textual data '''

        self.ln4 = nn.Linear(70, 64)
        self.ln5 = nn.Linear(70, 32)
        self.ln6 = nn.Linear(70, 16)
        self.ln7 = nn.Linear(70, 8)
        self.ln8 = nn.Linear(70, 2)

        # self.conv1 = nn.Conv2d(
        #     in_channels=70, out_channels=64, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(
        #     in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # self.conv8 = nn.Conv2d(
        #     in_channels=64, out_channels=1,  kernel_size=3, padding=1)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.bn6 = nn.BatchNorm2d(64)

        # self.dataset_dir = "/home/mdsamiul/InSAR-Coding/data/BSDS300/images"
        # self.train_acc = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()
    # breakpoint()

    def forward(self, x, y):  # forward propagation

        in_data = F.relu(self.conv1(x))

        in_data = F.relu(self.conv2(in_data))
        in_data = F.relu(self.conv3(in_data))
        in_data = F.relu(self.conv4(in_data))
        in_data = F.relu(self.conv5(in_data))
        in_data = F.relu(self.conv6(in_data))
        in_data = F.relu(self.conv7(in_data))

        # in_data = F.relu(self.bn1(self.conv2(in_data)))
        # in_data = F.relu(self.bn2(self.conv3(in_data)))
        # in_data = F.relu(self.bn3(self.conv4(in_data)))
        # in_data = F.relu(self.bn4(self.conv5(in_data)))
        # in_data = F.relu(self.bn5(self.conv6(in_data)))
        # in_data = F.relu(self.bn6(self.conv7(in_data)))
        in_data = self.conv8(in_data)

        in_data = in_data.reshape(in_data.shape[0], -1)

        in_data = F.relu(self.ln1(in_data))
        in_data = F.relu(self.ln2(in_data))

        tab = F.relu(self.ln4(y))
        tab = F.relu(self.ln5(tab))
        tab = F.relu(self.ln6(tab))
        tab = F.relu(self.ln7(tab))
        tab = F.relu(self.ln8(tab))

        x = torch.cat((in_data, tab), dim=1)
        x = F.relu(x)

        # tab = self.relu(tab)

        # y = residual + x

        # return y
        return x

    def training_step(self, batch, batch_idx):

        input_filt, coh, ddays, bperps, mr = batch
        # x=batch.float() dont work

        y_pred = self(input_filt, ddays)  # previous code

        # y_pred = torch.flatten(self(input_filt, ddays))

        mse = nn.MSELoss()
        loss = mse(mr, y_pred)

        tensorboard_logs = {'train_loss': loss}
        # self.train_acc(out, y)
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        return {'loss': loss, 'log': tensorboard_logs}

    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     loss = F.cross_entropy(y_hat, y)
    #     self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        mse = nn.MSELoss()
        loss = mse(y, out)

        tensorboard_logs = {'test_loss': loss}
        return {'test_loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):

        self.train_dataset = SpatialTemporalDataset(filt_dir='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/',
                                                    filt_ext='.diff.orb.statm_cor.natm.filt',
                                                    coh_dir='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/ifg_hr/',
                                                    coh_ext='.diff.orb.statm_cor.natm.filt.coh',
                                                    width=1500,
                                                    height=1500,
                                                    ref_mr_path='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/def_fit_cmpy',
                                                    ref_he_path='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/miami.tsx.sm_dsc.740.304.1500.1500/fit_hr/hgt_fit_m',
                                                    patch_size=28,
                                                    stride=0.5
                                                    )

        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=4,
                                      shuffle=True,
                                      num_workers=4,
                                      )

        return train_dataloader


if __name__ == '__main__':

    print("\n Hurray!!! \n \n Model is Running Perfectly \n ")
