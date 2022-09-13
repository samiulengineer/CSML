import re
import numpy as np
import matplotlib.pyplot as plt

from numba import jit
from datetime import datetime
from mrc_insar_common.data import data_reader



@jit(nopython=True)  # to fast the code like C. nopython=true is used for not falling back numba
def wrap(phase):
    return np.angle(np.exp(1j * phase)) # return angle value in radians and np.exp returns only complex value


def get_delta_days(date_string):
    date_format = "%Y%m%d"
    tokens = re.split("_|\.", date_string)
    date1 = datetime.strptime(tokens[0], date_format)
    date2 = datetime.strptime(tokens[1], date_format)
    delta_days = np.abs((date2 - date1).days)
    return delta_days


def gen_sim_mr_he(num_of_mr, num_of_he, min_mr, max_mr, min_he, max_he):
    sim_signals = np.zeros([num_of_he * num_of_mr, 2])
    fmrs = np.random.uniform(min_mr, max_mr, num_of_mr).round(1)
    fhes = np.random.uniform(min_he, max_he, num_of_he).round(2)
    idx = 0
    for mr in fmrs:
        for he in fhes:
            sim_signals[idx] = [mr, he]
            idx += 1
    return sim_signals


def gen_sim_3d(mr,
               he,
               stack_length,
               bperp_scale=2000.,
               dday_stepmax=4,
               dday_scale=11,
               conv1_scale=-0.0000573803,
               conv1_shift=-0.0110171730107,
               conv2_scale=-0.00073405573,
               conv2_shift=-0.00086772422789899997):

    """gen_sim_3d.
    Generate simulated unwraped recon phase with given mr and he
    
    Args:
        mr: motion rate with shape [H, W]
        he: height error with shape [H, W]
        stack_length: length of stack size
    """

    ddays = np.random.randint(low=1, high=dday_stepmax, size=stack_length) * dday_scale  # shape: [stack_length]
    bperps = (np.random.rand(stack_length) -0.5) * bperp_scale  # shape: [stack_length]
    conv1 = np.random.rand() * (conv1_scale) + conv1_shift #random single number
    conv2 = np.random.rand() * (conv2_scale) + conv2_shift #random simgle number
    unwrap_recon_phase = conv1 * ddays * (np.expand_dims(mr, -1)) + conv2 * bperps * (np.expand_dims(he, -1))  # shape: [H, W, stack_length]
    # reshapped_unwrap_recon_phase = unwrap_recon_phase.reshape(stack_length,np.expand_dims(mr, -1),np.expand_dims(mr, -1))
    return unwrap_recon_phase, ddays, bperps, conv1, conv2 # end of function
    # return reshapped_unwrap_recon_phase, ddays, bperps, conv1, conv2 # end of function





''' Visualize the output of the above function '''


if __name__ == "__main__":


    # load 3vG mr and he
    mr_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/elabra.tsx.sm_dsc.4624.118.1500.1500/fit_hr/def_fit_cmpy' 
    he_path = '/mnt/hdd1/3vG_data/3vg_parameter_fitting_data/elabra.tsx.sm_dsc.4624.118.1500.1500/fit_hr/hgt_fit_m' 

    mr = data_reader.readBin(mr_path, 1500, 'float') #shape(1500,1500)
    he = data_reader.readBin(he_path, 1500, 'float')
    
    

    SIM_STACK_LEN = 10 

    # check gen_sim_3d interface and implementation at: https://github.com/UAMRC-3vG/MRC-InSAR-Common/blob/main/mrc_insar_common/util/sim.py#L11
    # unwrap_phase, ddays, bperps, conv1, conv2 = sim.gen_sim_3d(mr, he, SIM_STACK_LEN)
    unwrap_phase, ddays, bperps, conv1, conv2 = gen_sim_3d(mr, he, SIM_STACK_LEN)


    # vsulize sample patchs in a batch
    print(wrap(unwrap_phase).shape)
    print(ddays.shape)
    print(bperps.shape)
    print(conv1)
    print(conv2)

    swapaxes_unwrap = np.swapaxes(unwrap_phase,0,2)
    reshape_unwrap = unwrap_phase.reshape(SIM_STACK_LEN,1500,1500)
    transpose_unwrap = np.transpose(unwrap_phase,(2,0,1))

    print(swapaxes_unwrap.shape)
    print('Wrap recon phase Shape = {}'.format(reshape_unwrap.shape))
    
    
    print('\n Wrap Reconstruction Phase (H, W, N)')
    fig, axs = plt.subplots(1,4, figsize=(9,2))
    for i in range(SIM_STACK_LEN): # size of stack
        # im = axs[i].imshow(wrap(unwrap_phase[:,:, i]), cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='None')
        im = axs[i].imshow(wrap(unwrap_phase[:,:, i]), cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()
    # plt.savefig('/home/niloy/Desktop/ps.mr.png',dpi=200,bbox_inches='tight')


    print('\n using swap axes (N, H, W)')
    fig, axs = plt.subplots(1,4, figsize=(9,2))
    for i in range(SIM_STACK_LEN): # size of stack
        # im = axs[i].imshow(wrap(unwrap_phase[:,:, i]), cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='None')
        im = axs[i].imshow(wrap(swapaxes_unwrap[i,:,:]), cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()


    print('\n using reshape (N, H, W)')
    fig, axs = plt.subplots(1,4, figsize=(9,2))
    for i in range(SIM_STACK_LEN): # size of stack
        # im = axs[i].imshow(wrap(unwrap_phase[:,:, i]), cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='None')
        im = axs[i].imshow(wrap(reshape_unwrap[i,:,:]), cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()


    print('\n using transpose (N, H, W)')
    fig, axs = plt.subplots(1,4, figsize=(9,2))
    for i in range(SIM_STACK_LEN): # size of stack
        # im = axs[i].imshow(wrap(unwrap_phase[:,:, i]), cmap='jet', vmin=-np.pi, vmax=np.pi, interpolation='None')
        im = axs[i].imshow(wrap(transpose_unwrap[i,:,:]), cmap='jet', vmin=-np.pi, vmax=np.pi)
        fig.colorbar(im, ax=axs[i], shrink=0.6, pad=0.05, fraction=0.046) 
        if i == 3: 
            break
    fig.tight_layout()
    plt.show()


'''
example of diferences between reshape,swapaxes
swapaxes works without changing value but reshape not

x = np.array([[[0,1,3],[2,3,3]],[[4,5,6],[6,7,8]]])

print('main')
print(x)
print(x.shape)
y = x.reshape(3,2,2)
print('reshape')
print(y)
print(y.shape)

print(np.swapaxes(y,0,1).shape)
z = np.swapaxes(x,0,2)
print('swap')
print(z)
print(z.shape) '''