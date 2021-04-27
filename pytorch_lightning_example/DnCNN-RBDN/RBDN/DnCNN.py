import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from DnCNN_Dataloader import NoisyDataset


''' Hyperparameter '''

# learning_rate = 1e-3
# hidden_dim = 128
# batch_size = 32


def dataset_imshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    image = (image + 1) / 2
    image[image < 0] = 0
    image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')

    return h


class DnCNN(pl.LightningModule):

    def __init__(self):
        super(DnCNN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,  out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=3,  kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64, 64)
        self.bn2 = nn.BatchNorm2d(64, 64)
        self.bn3 = nn.BatchNorm2d(64, 64)
        self.bn4 = nn.BatchNorm2d(64, 64)
        self.bn5 = nn.BatchNorm2d(64, 64)
        self.bn6 = nn.BatchNorm2d(64, 64)

        self.dataset_dir = "./data/BSDS300/images"
        # self.train_acc = pl.metrics.Accuracy()
        # self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):  # forward propagation
        in_data = F.relu(self.conv1(x))
        in_data = F.relu(self.bn1(self.conv2(in_data)))
        in_data = F.relu(self.bn2(self.conv3(in_data)))
        in_data = F.relu(self.bn3(self.conv4(in_data)))
        in_data = F.relu(self.bn4(self.conv5(in_data)))
        in_data = F.relu(self.bn5(self.conv6(in_data)))
        in_data = F.relu(self.bn6(self.conv7(in_data)))
        residual = self.conv8(in_data)

        y = residual + x

        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        mse = nn.MSELoss()
        loss = mse(y, out)

        # tensorboard_logs = {'train_loss': loss}
        # self.train_acc(out, y)
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=False)
        # return {'loss': loss, 'log': tensorboard_logs}
        return loss

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

        # tensorboard_logs = {'test_loss': loss}
        # return {'test_loss': loss, 'log': tensorboard_logs}
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    ''' dataset preparation '''

    # def setup(self, stage):

    #     dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
    #     self.mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
    #     self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(NoisyDataset(self.dataset_dir),
                          batch_size=20,
                          num_workers=4)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size)

    def test_dataloader(self):
        return DataLoader(NoisyDataset("./data/BSDS300/images", mode='test', img_size=(320, 320)), num_workers=4)


def main():

    # ------------
    # model
    # ------------

    model = DnCNN()

    # checkpoint_callback = ModelCheckpoint(filepath='./checkpoints/',
    #                                         save_top_k=1,
    #                                         monitor='loss',
    #                                         verbose=True)

    # logger = TensorBoardLogger( "lightning_logs",
    #                             name = "DnCNN"
    #                             )

    # ------------
    # training
    # ------------

    trainer = pl.Trainer(max_epochs=10,
                         #   fast_dev_run = False,
                         #  gpus=1,
                         #   logger = logger
                         )
    trainer.fit(model)
    pretrained_model = DnCNN.load_from_checkpoint(
        "./checkpoints/epoch=29.ckpt")

    # ------------
    # testing
    # ------------

    test_set = trainer.test(model)
    with torch.no_grad():
        out = pretrained_model(test_set[2][0].unsqueeze(0))
    fig, axes = plt.subplots(ncols=2)
    dataset_imshow(test_set[2][0], ax=axes[0])
    axes[0].set_title('Noisy')
    dataset_imshow(out[0], ax=axes[1])
    axes[1].set_title('Clean')
    print(f'image size is {out[0].shape}.')

    # ------------
    # testing
    # ------------


if __name__ == '__main__':
    main()
