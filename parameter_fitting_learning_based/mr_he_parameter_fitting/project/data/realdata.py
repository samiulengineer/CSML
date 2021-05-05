import torch
from mrc_insar_common.data import data_reader
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from mrc_insar_common.util.sim import gen_sim_3d
from mrc_insar_common.util.utils import wrap
import numpy as np
import logging
import glob
import tqdm
from datetime import datetime
import re
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf


logger = logging.getLogger(__name__)


def get_delta_days(date_string):
    date_format = "%Y%m%d"
    tokens = re.split("_|\.", date_string)
    date1 = datetime.strptime(tokens[0], date_format)
    date2 = datetime.strptime(tokens[1], date_format)
    delta_days = np.abs((date2 - date1).days)
    return delta_days


class SpatialTemporalTrainDataset(Dataset):

    def __init__(self,
                 filt_dir,
                 filt_ext,
                 bperp_dir,
                 bperp_ext,
                 coh_dir,
                 coh_ext,
                 conv1,
                 conv2,
                 width,
                 height,
                 ref_mr_path,
                 ref_he_path,
                 name,
                 sim,
                 patch_size=38,
                 stride=0.5):
        self.filt_paths = sorted(
            glob.glob('{}/*{}'.format(filt_dir, filt_ext)))
        self.bperp_paths = sorted(
            glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path
        self.conv1 = conv1
        self.conv2 = conv2
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.stride = stride
        self.name = name
        self.sim = sim

        self.stack_size = len(self.filt_paths)

        self.ddays = np.zeros(self.stack_size)
        self.bperps = np.zeros(self.stack_size)

        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(
                bperp_path, 1, 'float')[0][0]

        logger.info(f'stack {self.name} buffer data loaded')

        self.all_sample_coords = [(row_idx, col_idx)
                                  for row_idx in range(0, self.height - self.patch_size - 1, int(self.patch_size * stride))
                                  for col_idx in range(0, self.width - self.patch_size - 1, int(self.patch_size * stride))]

    def __len__(self):
        return len(self.all_sample_coords)

    def __getitem__(self, idx):
        coord = self.all_sample_coords[idx]

        mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(
            coord[0], coord[1], self.patch_size, self.patch_size))
        he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(
            coord[0], coord[1], self.patch_size, self.patch_size))

        # [N, h ,w] for a single training sample,
        filt_input = np.zeros(
            [self.stack_size, self.patch_size, self.patch_size])
        # [N, h ,w] for a single training sample
        coh_input = np.zeros(
            [self.stack_size, self.patch_size, self.patch_size])

        if (self.sim):
            unwrap_sim_phase, sim_ddays, sim_bperps, sim_conv1, sim_conv2 = gen_sim_3d(
                mr_target, he_target, self.stack_size)  # [h, w, N] unwrapped phase
            wrap_sim_phase = wrap(unwrap_sim_phase)
            filt_input = np.transpose(wrap_sim_phase, [2, 0, 1])  # [N, h, w]
            coh_input += 1  # coh is 1 for simuluation data
            ddays = sim_ddays
            bperps = sim_bperps
            conv1 = sim_conv1
            conv2 = sim_conv2

        else:
            ddays = self.ddays
            bperps = self.bperps
            conv1 = self.conv1
            conv2 = self.conv2
            for i in range(self.stack_size):
                filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(
                    coord[0], coord[1], self.patch_size, self.patch_size)))

                coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(
                    coord[0], coord[1], self.patch_size, self.patch_size))

        return {
            'phase': filt_input.astype(np.float32),
            'coh': coh_input.astype(np.float32),
            'mr': np.expand_dims(mr_target, 0).astype(np.float32),
            'he': np.expand_dims(he_target, 0).astype(np.float32),
            'ddays': ddays.astype(np.float32),
            'bperps': bperps.astype(np.float32),
            'conv1': float(conv1),
            'conv2': float(conv2)
        }


class SpatialTemporalTestDataset(Dataset):

    def __init__(self,
                 filt_dir,
                 filt_ext,
                 bperp_dir,
                 bperp_ext,
                 coh_dir,
                 coh_ext,
                 conv1,
                 conv2,
                 width,
                 height,
                 ref_mr_path,
                 ref_he_path,
                 name,
                 roi):
        self.filt_paths = sorted(
            glob.glob('{}/*{}'.format(filt_dir, filt_ext)))
        self.bperp_paths = sorted(
            glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path
        self.conv1 = conv1
        self.conv2 = conv2
        self.width = width
        self.height = height
        self.name = name
        self.roi = roi

        self.stack_size = len(self.filt_paths)

        self.ddays = np.zeros(self.stack_size)
        self.bperps = np.zeros(self.stack_size)

        coord = self.roi

        self.roi_height = self.roi[2] - self.roi[0]
        self.roi_width = self.roi[3] - self.roi[1]

        self.mr_target = data_reader.readBin(self.ref_mr_path, self.width, 'float', crop=(
            coord[0], coord[1], self.roi_height, self.roi_width))
        self.mr_target = np.expand_dims(self.mr_target, 0)
        self.he_target = data_reader.readBin(self.ref_he_path, self.width, 'float', crop=(
            coord[0], coord[1], self.roi_height, self.roi_width))
        self.he_target = np.expand_dims(self.he_target, 0)

        # [N, h ,w] for a single training sample,
        self.filt_input = np.zeros(
            [self.stack_size, self.roi_height, self.roi_width])
        # [N, h ,w] for a single training sample
        self.coh_input = np.zeros(
            [self.stack_size, self.roi_height, self.roi_width])

        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(
                bperp_path, 1, 'float')[0][0]

            self.filt_input[idx] = np.angle(data_reader.readBin(
                self.filt_paths[idx], self.width, 'floatComplex', crop=(coord[0], coord[1], self.roi_height, self.roi_width)))

            self.coh_input[idx] = data_reader.readBin(self.coh_paths[idx], self.width, 'float', crop=(
                coord[0], coord[1], self.roi_height, self.roi_width))
        logger.info(f'stack {self.name} buffer data loaded')

    def __len__(self):
        return 1  # single ROI for each stack

    def __getitem__(self, idx):

        return {
            'phase': self.filt_input.astype(np.float32),
            'coh': self.coh_input.astype(np.float32),
            'mr': self.mr_target.astype(np.float32),
            'he': self.he_target.astype(np.float32),
            'ddays':
                self.
                ddays.astype(np.float32),
            'bperps':
                self.bperps.astype(np.float32),    #
            'conv1': float(self.conv1),
            'conv2': float(self.conv2)
        }


class RealInSARDataModule(LightningDataModule):
    def __init__(self, train_stacks, val_stacks, num_workers, patch_size: int = 21, batch_size: int = 32, train_stride=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_workers = num_workers
        self.patch_size = patch_size
        self.batch_size = batch_size

        self.train_dataset = torch.utils.data.ConcatDataset([SpatialTemporalTrainDataset(
            **stack, patch_size=patch_size, stride=train_stride) for stack in train_stacks])
        self.val_dataset = torch.utils.data.ConcatDataset(
            [SpatialTemporalTestDataset(**stack) for stack in val_stacks])
        logger.info(
            f'len of train examples {len(self.train_dataset)}, len of val examples {len(self.val_dataset)}')

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=self.num_workers,
                                                   drop_last=True,
                                                   pin_memory=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=self.num_workers,
                                                 drop_last=True,
                                                 pin_memory=True)
        return val_loader


if __name__ == "__main__":

    train_dataset = torch.utils.data.ConcatDataset(SpatialTemporalTrainDataset(filt_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
                                                                               filt_ext=".diff.orb.statm_cor.natm.filt",
                                                                               coh_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
                                                                               coh_ext=".diff.orb.statm_cor.natm.filt.coh",
                                                                               bperp_dir="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/",
                                                                               bperp_ext=".bperp",
                                                                               ref_mr_path="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/fit_hr/def_fit_cmpy",
                                                                               ref_he_path="/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/fit_hr/hgt_fit_m",
                                                                               conv1=-0.0110745533168,
                                                                               conv2=-0.00122202886324,
                                                                               width=1500,
                                                                               height=1500,
                                                                               sim=True,
                                                                               name="cortez"
                                                                               ))
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=4,
                                  drop_last=True,
                                  pin_memory=True
                                  )

    for batch_idx, batch in enumerate(train_dataset):

        ''' if we want to return not in dictionary we can check in this way '''

        # input_filt, coh, ddays, bperps, mr, he = batch
        [B, N] = batch['ddays'].shape

        # print(input_filt.shape)

        ''' if we want to return in dictionary we can check in this way '''

        print('Batch Index \t = {}'.format(batch_idx))
        # print('Input Type \t = {}'.format(batch['input'].dtype))
        # print('Input Shape \t = {}'.format(batch['input'].shape))
        print('Coh Shape \t = {}'.format(batch['coh'].shape))
        print('mr Shape \t = {}'.format(batch['mr'].shape))
        # print(batch['mr']) # to check the output of mr
        print('he Shape \t = {}'.format(batch['he'].shape))
        print('ddays Shape \t = {}'.format(batch['ddays'].shape))
        print('bperps Shape \t = {}'.format(batch['bperps'].shape))
        print('conv1 shape \t = {}'.format(batch['conv1'].shape))
        # print('Wrap recon phase shape = {}'.format(
        #     batch['wrap_recon_phase'].shape))

        # print(np.angle(1*np.exp(batch['input'][0][0][0]) - (batch['wrap_recon_phase'][0][0][0])))

        break

    ''' vsulize sample patches in a batch from train dataset'''

    print('\n ---------------- mr ---------------- \n')
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    input_shape = batch['mr'][0].shape  # first training example
    # print (batch['mr'][0][0].shape)
    for i in range(input_shape[2]):  # size of stack
        im = axs[i].imshow(batch['mr'][0][0], cmap='jet',
                           vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 1:
            break
    fig.tight_layout()
    plt.suptitle("Motion Rate Map", fontsize=14)
    plt.savefig("/mnt/hdd1/mdsamiul/mr_real.png")
    plt.show()

    print('\n ---------------- he ---------------- \n')
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    input_shape = batch['he'][0].shape  # first training example
    # print (batch['he'][0][0].shape)
    for i in range(input_shape[2]):  # size of stack
        im = axs[i].imshow(batch['he'][0][0], cmap='jet',
                           vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 1:
            break
    fig.tight_layout()
    plt.suptitle("Height Error Map", fontsize=14)
    plt.savefig("/mnt/hdd1/mdsamiul/he_real.png")
    plt.show()
