from mrc_insar_common.data import data_reader
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn


import matplotlib.pyplot as plt
import numpy as np
import glob
import tqdm
import numpy as np

import torch.nn.functional as F

import matplotlib.pyplot as plt
from mrc_insar_common.data import data_reader
from utils import gen_sim_3d, wrap, get_delta_days


class SpatialTemporalDataset(Dataset):  # inherit the dataset class

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
                 patch_size=38,
                 stride=0.5
                 ):

        # provide a list of sorted data file path of filt_ext fometed file
        self.filt_paths = sorted(
            glob.glob('{}/*{}'.format(filt_dir, filt_ext)))

        # provide a list of sorted data file path of bperp_ext fometed file
        self.bperp_paths = sorted(
            glob.glob('{}/*{}'.format(bperp_dir, bperp_ext)))
        # rovide a list of sorted data file path of coh_ext fometed file
        self.coh_paths = sorted(glob.glob('{}/*{}'.format(coh_dir, coh_ext)))
        self.ref_mr_path = ref_mr_path
        self.ref_he_path = ref_he_path

        # self.ref_mr_path = data_reader.readBin(ref_mr_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))
        # crop function mainly crop from the coord coordinator like (x,y) = (904,532) and crop patch_size * patch_size
        # self.ref_he_path = data_reader.readBin(ref_he_path, self.width, 'float', crop=(coord[0], coord[1], self.patch_size, self.patch_size))
        self.conv1 = conv1
        self.conv2 = conv2
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.stride = stride

        # find out the lenght of the sorted self.filt_path                                                                                  self.stack_size)
        self.stack_size = len(self.filt_paths)

        self.ddays = np.zeros(self.stack_size)  # make a 1d list 0 of length of
        self.bperps = np.zeros(self.stack_size)

        # tqdm >>it's just add a animation of loading
        for idx in tqdm.tqdm(range(self.stack_size)):
            # read delta days
            # taking every element in the list one by one
            bperp_path = self.bperp_paths[idx]
            date_string = bperp_path.split('/')[-1].replace(bperp_ext, "")
            # split the the element in the list with respect to "/" [-1] represent the last element in the splited list
            # then replace the extention(bparp_ext) part with space

            delta_day = get_delta_days(date_string)
            self.ddays[idx] = delta_day

            # read bperp
            self.bperps[idx] = data_reader.readBin(
                bperp_path, 1, 'float')[0][0]
        # print(self.bperps)

        self.all_sample_coords = [(row_idx, col_idx)
                                  for row_idx in range(0, self.height - self.patch_size - 1, int(self.patch_size * stride))
                                  # range(start,end,skip)
                                  # 0,1500-28-1, int(28*0.5)
                                  # 0,1471,14
                                  # to see the example please run this code
                                  # x =[(i,j) for i in range(0,1471,14) for j in range(0,1471,14)]
                                  # print(x)
                                  for col_idx in range(0, self.width - self.patch_size - 1, int(self.patch_size * stride))]

    def __len__(self):
        return len(self.all_sample_coords)

    def __getitem__(self, idx):
        # print (coord) # print coord to check the output of the coord
        coord = self.all_sample_coords[idx]
        # crop function mainly crop from the coord coordinator like (x,y) = (904,532) and crop patch_size * patch_size
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

        self.unwrap_recon_phase, self.ddays_sim, self.bperps_sim, self.conv1_sim, self.conv2_sim = gen_sim_3d(
            mr_target, he_target, self.stack_size)

        wrap_recon_phase = np.transpose(
            wrap(self.unwrap_recon_phase), (2, 0, 1))

        for i in range(self.stack_size):
            # !! here is an example that only uses phase information
            # MRC InSAR Library - https://pypi.org/project/MRC-InSAR-Common/  follow this link  for datareader of readbin
            filt_input[i] = np.angle(data_reader.readBin(self.filt_paths[i], self.width, 'floatComplex', crop=(
                coord[0], coord[1], self.patch_size, self.patch_size)))

            coh_input[i] = data_reader.readBin(self.coh_paths[i], self.width, 'float', crop=(
                coord[0], coord[1], self.patch_size, self.patch_size))

        ''' return in dictionary format if we want to plot all the data after Dataloader '''

        return{
            'input': filt_input,  # 3D data
            'coh': coh_input,  # feature 3D
            # label expand dims is used for convert the array in matrix
            'mr': np.expand_dims(mr_target, 0),
            # axis = 0 means increase in column, axis = 1 means increase in row
            'he': np.expand_dims(he_target, 0),  # same as mr
            # ddays and bperps are shared for all training samples in a stack, it can be used in a more effecient way, here is just an example
            'ddays_sim': self.ddays_sim,
            'ddays': self.ddays,
            'bperps_sim': self.bperps_sim,   # feature single value
            'bperps': self.bperps,
            'conv1_sim': self.conv1_sim,
            'conv1': self.conv1,
            'conv2_sim': self.conv2_sim,
            'conv2': self.conv2,
            'unwrap_recon_phase': self.unwrap_recon_phase,
            'wrap_recon_phase': wrap_recon_phase
        }

        ''' return in dictionary format if we want to plot all the data after Dataloader '''

        ''' conv2d can not take float64 tensor, so here we decided to convert it into
            float32 array before passing through Dataloader '''

        # input_filt = np.float32(filt_input)
        # coh = np.float32(coh_input)
        # ddays = np.float32(self.ddays)
        # bperps = np.float32(self.bperps)

        # mr = np.float32(np.expand_dims(mr_target, 0))
        # he = np.float32(np.expand_dims(he_target, 0))

        # return input_filt, coh, ddays, bperps, mr, he


if __name__ == "__main__":

    train_dataset = SpatialTemporalDataset(filt_dir='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/bagdad.tsx.sm_dsc.1700.400.1500.1500/ifg_hr/',
                                           filt_ext='.diff.orb.statm_cor.natm.filt',
                                           bperp_dir='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/bagdad.tsx.sm_dsc.1700.400.1500.1500/ifg_hr/',
                                           bperp_ext='.bperp',
                                           coh_dir='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/cortez.tsx.sm_dsc.3100.500.1500.1500/ifg_hr/',
                                           coh_ext='.diff.orb.statm_cor.natm.filt.coh',
                                           conv1=-0.0110745533168,
                                           conv2=-0.00134047881374,
                                           width=1500,
                                           height=1500,
                                           ref_mr_path='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/bagdad.tsx.sm_dsc.1700.400.1500.1500/fit_hr/def_fit_cmpy',
                                           ref_he_path='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/bagdad.tsx.sm_dsc.1700.400.1500.1500/fit_hr/hgt_fit_m',
                                           patch_size=28,
                                           stride=0.5
                                           )

    # test_dataset = SpatialTemporalDataset(filt_dir='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/chino.tsx.sm_dsc.1047.1859.1500.1500/ifg_hr',
    #                                       filt_ext='.diff.orb.statm_cor.natm.filt',
    #                                       coh_dir='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/chino.tsx.sm_dsc.1047.1859.1500.1500/ifg_hr',
    #                                       coh_ext='.diff.orb.statm_cor.natm.filt.coh',
    #                                       width=1500,
    #                                       height=1500,
    #                                       ref_mr_path='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/chino.tsx.sm_dsc.1047.1859.1500.1500/fit_hr/def_fit_cmpy',
    #                                       ref_he_path='/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/chino.tsx.sm_dsc.1047.1859.1500.1500/fit_hr/hgt_fit_m',
    #                                       patch_size=28,
    #                                       stride=0.5
    #                                       )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=4,
                                  shuffle=True,
                                  num_workers=4,
                                  )

    # test_dataloader = DataLoader(test_dataset,
    #                              batch_size=4,
    #                              shuffle=True,
    #                              num_workers=4,
    #                              )

    ''' Output & Visualize the training data '''

    print('train_dataset length {}'.format(len(train_dataset)))
    print('type of train_dataset {}'.format(type(train_dataset)))

    for batch_idx, batch in enumerate(train_dataloader):

        ''' if we want to return not in dictionary we can check in this way '''

        # input_filt, coh, ddays, bperps, mr, he = batch
        [B, N] = batch['ddays'].shape

        # print(input_filt.shape)

        ''' if we want to return in dictionary we can check in this way '''

        print('Batch Index \t = {}'.format(batch_idx))
        print('Input Type \t = {}'.format(batch['input'].dtype))
        print('Input Shape \t = {}'.format(batch['input'].shape))
        print('Coh Shape \t = {}'.format(batch['coh'].shape))
        print('mr Shape \t = {}'.format(batch['mr'].shape))
        # print(batch['mr']) # to check the output of mr
        print('he Shape \t = {}'.format(batch['he'].shape))
        print('ddays Shape \t = {}'.format(batch['ddays'].shape))
        print('bperps Shape \t = {}'.format(batch['bperps'].shape))
        print('conv1 shape \t = {}'.format(batch['conv1'].shape))
        print('Wrap recon phase shape = {}'.format(
            batch['wrap_recon_phase'].shape))

        # print(np.angle(1*np.exp(batch['input'][0][0][0]) - (batch['wrap_recon_phase'][0][0][0])))

        break

    ''' vsulize sample patches in a batch from train dataset'''

    print('\n ---------------- input ---------------- \n')
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    input_shape = batch['input'][0].shape  # first training example
    # print (input_shape[0])
    for i in range(input_shape[2]):  # size of stack
        im = axs[i].imshow(batch['input'][0][i], cmap='jet',
                           vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 3:
            break
    fig.tight_layout()
    plt.suptitle("Real Input Interferogram", fontsize=14)
    plt.savefig("/mnt/hdd1/mdsamiul/input_real.png", dpi=500)
    plt.show()

    print('\n ---------------- coh ---------------- \n')
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    coh_shape = batch['coh'][0].shape  # first training example
    for i in range(coh_shape[2]):  # size of stack
        im = axs[i].imshow(batch['coh'][0][i], cmap='gray', vmin=0, vmax=1)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 3:
            break
    fig.tight_layout()
    plt.suptitle("Real Coherence", fontsize=14)
    plt.savefig("/mnt/hdd1/mdsamiul/coh_real.png")
    plt.show()

    print('\n ---------------- mr ---------------- \n')
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    input_shape = batch['mr'][0].shape  # first training example
    # print (batch['mr'][0][0].shape)
    for i in range(input_shape[2]):  # size of stack
        im = axs[i].imshow(batch['mr'][i][0], cmap='jet',
                           vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 3:
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
        im = axs[i].imshow(batch['he'][i][0], cmap='jet',
                           vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 3:
            break
    fig.tight_layout()
    plt.suptitle("Height Error Map", fontsize=14)
    plt.savefig("/mnt/hdd1/mdsamiul/he_real.png")
    plt.show()

    print('\n ---------------- wrap_recon_phase ---------------- \n')
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    input_shape = batch['wrap_recon_phase'][0].shape  # first training example
    # print (input_shape[0])
    for i in range(input_shape[2]):  # size of stack
        im = axs[i].imshow(batch['wrap_recon_phase'][0][i],
                           cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 3:
            break
    fig.tight_layout()
    plt.suptitle("Simulation Input", fontsize=14)
    plt.savefig("/mnt/hdd1/mdsamiul/input_wrap_sim.png")
    plt.show()

    print('\n ---------------- Difference in Real & Simulation Input ---------------- \n')
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    # first training example
    input_shape = batch['unwrap_recon_phase'][0].shape
    for i in range(input_shape[2]):  # size of stack
        im = axs[i].imshow(np.angle(1*np.exp(batch['input'][0][i]) -
                                    (batch['wrap_recon_phase'][0][i])), cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046)
        if i == 3:
            break
    fig.tight_layout()
    plt.suptitle("Difference in Real & Simulation Input", fontsize=14)
    plt.savefig("/mnt/hdd1/mdsamiul/dif_real_sim.png")
    plt.show()

    print('\n ---------------- Real ddays ---------------- \n')

    s = range(1, N+1)
    sequence_of_numbers = [number for number in s]

    plt.figure(figsize=(15, 10))
    plt.plot(sequence_of_numbers, batch['ddays'][0].tolist())
    plt.title('Real ddays', fontsize=18)
    plt.xlabel('Stack Size', fontsize=18)
    plt.ylabel('ddays', fontsize=18)
    plt.savefig("/mnt/hdd1/mdsamiul/ddays.png")
    plt.show()

    print('\n ---------------- Simulation ddays ---------------- \n')

    plt.figure(figsize=(15, 10))
    plt.plot(sequence_of_numbers, batch['ddays_sim'][0].tolist())
    plt.title('Simulation ddays', fontsize=18)
    plt.xlabel('Stack Size', fontsize=18)
    plt.ylabel('ddays_sim', fontsize=18)
    plt.savefig("/mnt/hdd1/mdsamiul/ddays_sim.png")
    plt.show()

    print('\n ---------------- Real bperps ---------------- \n')

    plt.figure(figsize=(15, 10))
    plt.plot(sequence_of_numbers, batch['bperps'][0].tolist())
    plt.title('Real bperps', fontsize=18)
    plt.xlabel('Stack Size', fontsize=18)
    plt.ylabel('bperps', fontsize=18)
    plt.savefig("/mnt/hdd1/mdsamiul/bperps.png")
    plt.show()

    print('\n ---------------- Simulation bperps ---------------- \n')

    plt.figure(figsize=(15, 10))
    plt.plot(sequence_of_numbers, batch['bperps_sim'][0].tolist())
    plt.title('Simulation bperps', fontsize=18)
    plt.xlabel('Stack Size', fontsize=18)
    plt.ylabel('bperps_sim', fontsize=18)
    plt.savefig("/mnt/hdd1/mdsamiul/bperps_sim.png",
                dpi=100, bbox_inches='tight')
    plt.show()

    ''' show layer  '''
    ''' phase + coh in DnCNN '''

    # conv1 = nn.Conv2d(in_channels=135, out_channels=64,
    #                   kernel_size=3, padding=1)
    # conv2 = nn.Conv2d(in_channels=64, out_channels=64,
    #                   kernel_size=3, padding=1)
    # conv3 = nn.Conv2d(in_channels=64, out_channels=64,
    #                   kernel_size=3, padding=1)
    # conv4 = nn.Conv2d(in_channels=64, out_channels=64,
    #                   kernel_size=3, padding=1)
    # conv5 = nn.Conv2d(in_channels=64, out_channels=64,
    #                   kernel_size=3, padding=1)
    # conv6 = nn.Conv2d(in_channels=64, out_channels=64,
    #                   kernel_size=3, padding=1)
    # conv7 = nn.Conv2d(in_channels=64, out_channels=64,
    #                   kernel_size=3, padding=1)
    # conv8 = nn.Conv2d(in_channels=64, out_channels=2,
    #                   kernel_size=3, padding=1)
    # conv_concat = nn.Conv2d(
    #     in_channels=4, out_channels=1, kernel_size=3, padding=1)

    # bn1 = nn.BatchNorm2d(64)
    # bn2 = nn.BatchNorm2d(64)
    # bn3 = nn.BatchNorm2d(64)
    # bn4 = nn.BatchNorm2d(64)
    # bn5 = nn.BatchNorm2d(64)
    # bn6 = nn.BatchNorm2d(64)

    # cln1 = nn.Linear(4 * 28 * 28, 16)
    # batchnorm = nn.BatchNorm1d(16)
    # dropout = nn.Dropout2d(0.5)
    # cln2 = nn.Linear(16, 8)

    # ''' ddays + bperps in linear layer '''

    # ln1 = nn.Linear(135, 64)
    # ln2 = nn.Linear(64, 4)

    # ''' concat layer '''

    # ln_concat = nn.Linear(16, 2)

    # print('\n------------- phase -------------\n')

    # print("phase shape : {}" .format(input_filt.shape))

    # input_filt = conv1(input_filt)
    # print("phase conv1 shape : {}" .format(input_filt.shape))

    # input_filt = bn1(conv2(input_filt))
    # print("phase conv2 shape : {}" .format(input_filt.shape))

    # input_filt = bn2(conv3(input_filt))
    # print("phase conv3 shape : {}" .format(input_filt.shape))

    # input_filt = bn3(conv4(input_filt))
    # print("phase conv4 shape : {}" .format(input_filt.shape))

    # input_filt = bn4(conv5(input_filt))
    # print("phase conv5 shape : {}" .format(input_filt.shape))

    # input_filt = bn5(conv6(input_filt))
    # print("phase conv6 shape : {}" .format(input_filt.shape))

    # input_filt = bn6(conv7(input_filt))
    # print("phase conv7 shape : {}" .format(input_filt.shape))

    # input_filt = conv8(input_filt)
    # print("phase conv8 shape : {}" .format(input_filt.shape))

    # print('\n------------- coh -------------\n')

    # print("coh shape : {}" .format(coh.shape))

    # coh = conv1(coh)
    # print("coh conv1 shape : {}" .format(coh.shape))

    # coh = bn1(conv2(coh))
    # print("coh conv1 shape : {}" .format(coh.shape))

    # coh = bn2(conv3(coh))
    # print("coh conv1 shape : {}" .format(coh.shape))

    # coh = bn3(conv4(coh))
    # print("coh conv1 shape : {}" .format(coh.shape))

    # coh = bn4(conv5(coh))
    # print("coh conv1 shape : {}" .format(coh.shape))

    # coh = bn5(conv6(coh))
    # print("coh conv1 shape : {}" .format(coh.shape))

    # coh = bn6(conv7(coh))
    # print("coh conv1 shape : {}" .format(coh.shape))

    # coh = conv8(coh)
    # print("coh conv1 shape : {}" .format(coh.shape))

    # print('\n------------- concat_phase_coh -------------\n')

    # concat_phase_coh = F.relu(torch.cat((input_filt, coh), dim=1))
    # print("concat_phase_coh shape : {}" .format(concat_phase_coh.shape))

    # print('\n------------- concat_phase_coh_flatten -------------\n')

    # concat_phase_coh_flatten = concat_phase_coh.reshape(
    #     concat_phase_coh.shape[0], -1)
    # print("concat_phase_coh_flatten shape : {}" .format(
    #     concat_phase_coh_flatten.shape))

    # flattenL1 = F.relu(cln1(concat_phase_coh_flatten))
    # print("flattenL1 shape : {}" .format(
    #     flattenL1.shape))

    # flattenL2 = F.relu(cln2(flattenL1))
    # print("flattenL2 shape : {}" .format(
    #     flattenL2.shape))

    # # concat_phase_coh_last = F.relu(conv_concat(concat_phase_coh))
    # # print("concat_phase_coh_last shape : {}" .format(concat_phase_coh_last.shape))

    # print('\n------------- ddays -------------\n')
    # print("ddays shape: {}".format(ddays.shape))

    # ddays = F.relu(ln1(ddays))
    # print("ddays ln1 shape : {}" .format(ddays.shape))

    # ddays = F.relu(ln2(ddays))
    # print("ddays ln2 shape : {}" .format(ddays.shape))

    # print('\n------------- bperps -------------\n')

    # print("bperp shape: {}".format(bperps.shape))

    # bperps = F.relu(ln1(bperps))
    # print("bperp ln1 shape: {}".format(bperps.shape))

    # bperps = F.relu(ln2(bperps))
    # print("bperp ln2 shape: {}".format(bperps.shape))

    # print('\n------------- concat_ddays_bperps -------------\n')

    # concat_ddays_bperps = F.relu(torch.cat((ddays, bperps), dim=1))
    # print("concat_ddays_bperps shape: {}".format(concat_ddays_bperps.shape))

    # print('\n------------- concat_all -------------\n')

    # concat_all = F.relu(torch.cat((flattenL2, concat_ddays_bperps), dim=1))
    # print("concat_all shape: {}".format(concat_all.shape))

    # concat_all_layer = F.relu(ln_concat(concat_all))
    # print("concat_all_layer shape: {}".format(
    #     concat_all_layer.shape))

    # concat_all_layer_reshape = torch.reshape(concat_all_layer, [B, 2, 1, 1])
    # print("concat_all_layer_reshape shape: {}".format(concat_all_layer_reshape.shape))

    # ref_out = torch.cat([mr, he], 1)
    # print("hello mr he: {}".format(ref_out.shape))

    # # all_concat = torch.cat((concat_phase_coh_last, r))
    # # print("all_concat shape: {}".format(all_concat.shape))
