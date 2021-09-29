import pytorch_lightning as pl
# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint
from model import DnCNN
from dataloader_random import SpatialTemporalDataset


def main():

    # ------------
    # model
    # ------------

    model = DnCNN()

    # checkpoint_callback = ModelCheckpoint(filepath   = './parameter_fitting/',
    #                                       save_top_k = 1,
    #                                       monitor    = 'loss',
    #                                       verbose    = True)

    logger = TensorBoardLogger("lightning_logs",
                               name="DnCNN"
                               )

    # ------------
    # training
    # ------------

    trainer = pl.Trainer(max_epochs=10,
                         fast_dev_run=True,
                         gpus=2,
                         logger=logger
                         )

    trainer.fit(model)

    # pretrained_model = DnCNN.load_from_checkpoint("./parameter_fitting/DnCNN.ckpt")

    # ------------
    # testing
    # ------------

    # test_set = trainer.test(model)


if __name__ == '__main__':

    main()

    ''' plot single image after denoising '''

    # def dataset_imshow(image, ax=plt):
    #     image = image.to('cpu').numpy()
    #     image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    #     image = (image + 1) / 2
    #     image[image < 0] = 0
    #     image[image > 1] = 1
    #     h = ax.imshow(image)
    #     ax.axis('off')

    #     return h

    # with torch.no_grad():
    #     out = pretrained_model(test_set[2][0].unsqueeze(0))
    # fig, axes = plt.subplots(ncols=2)
    # dataset_imshow(test_set[2][0], ax=axes[0])
    # axes[0].set_title('Noisy')
    # dataset_imshow(out[0], ax=axes[1])
    # axes[1].set_title('Clean')
    # print(f'image size is {out[0].shape}.')
