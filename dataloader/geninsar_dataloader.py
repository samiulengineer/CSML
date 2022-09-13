# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 20:01:10 2021

@author: ashis
"""
import numpy as np
import json
import torch
import random
from mrc_insar_common.data import data_reader as data_read


patch_size = 21
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8,
          'drop_last': True}


'''

DataSet Class :
Input: Path for the Json file
Output: Batch_size x 21 x 21 x 2 
The patches will be produced randomly from the IFG path given in the JSON file.
"ifglist" in JSON will have list of IFG with each entry of the form
        {ifg_path} + 'width' + {width_of_ifg} + 'height' + {height_of_ifg}
'''
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, jsonfile):
        'Initialization'
        with open(jsonfile) as f: """open the  file  """
            data = json.load(f) #load all elment and convert it to usable py file
        data = json.loads(data) #make dictonary file  
        self.dataarray = data["ifglist"] #assain all the value inside the disctonary under the key of ifglist ,,it become a list 
 
  def __len__(self):
        ''' The total length of the dataset is number of {ifgs in the dataarray} * {Nummber of patches per ifg we want to train it on.}   '''
        'Denotes the total number of samples'
        return len(self.dataarray)*50*64


  def __getitem__(self, index):
        'Generates one sample of data'
        index = (index)// (50*64)
        x=self.dataarray[index]
        #\"/mnt/hdd1/3vG_data/caric_20170623_data/tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20150128_20150208.diff.orb_corwidth1000height1000\"
        ifgpath = x.split("width")[0] # Extracting ifgpath , width and height of the ifg from the string. 
# Extracting ifgpath , width and height of the ifg from the string. #when ever it encounter the "width" it will be split from there 
        '''exp h="do you knw hight and width is the boss of the town"
			ifgpath = h.split("width")[0].split("hight")[1]
			print(ifgpath) result : and 
                  
                  tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20140509_20140520.diff.orb_cor|width|1000height1000\
                        result = tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20140509_20140520.diff.orb_cor''''
        
        width = int(x.split("width")[-1].split("height")[0]) # think x=' tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20140509_20140520.diff.orb_cor|width|1000height1000'
                                                            # for this exp the outcome will be tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20140509_20140520.dif0f.orb_cor|width|[1000]height1000
                                                            #width = 1000

        height = int(x.split("width")[-1].split("height")[-1]) # think x=' tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20140509_20140520.diff.orb_cor|width|1000height1000'
                                                            # for this exp the outcome will be tsx_chuquicamata_sm_dsc_1000x1000_crop/ifg_fr/20140509_20140520.dif0f.orb_cor|width|1000height[1000]
                                                            #height= 1000

        #Getting random row and random column 
        random_row_idx = random.randint(0, height - patch_size - 1)
        random_col_idx = random.randint(0, width - patch_size - 1)        

        patch = data_read.readBin(ifgpath,width=width,dataType='floatComplex',crop=[random_col_idx,random_row_idx,patch_size,patch_size]) #A numpy array of the data


        patch = np.angle(patch)

        ifgreal   = np.cos(patch)
        ifgimag   = np.sin(patch)
        targreal  = np.copy(ifgreal)
        targimag  = np.copy(ifgimag)
        
        mid_pixel = patch_size // 2 #10 half point 
    
        ifgreal[mid_pixel,mid_pixel] = 0     
        ifgimag[mid_pixel,mid_pixel] = 0     # avoid learning identity mapping
        
        realpats = np.expand_dims(ifgreal,axis=-1)
        imagpats = np.expand_dims(ifgimag,axis=-1)
        reimpats = np.concatenate((realpats,imagpats),axis=-1)
        
        targreal = np.squeeze( targreal[mid_pixel,mid_pixel] )
        targimag = np.squeeze( targimag[mid_pixel,mid_pixel] )
        targpats = np.concatenate((np.expand_dims(targreal,axis=-1),np.expand_dims(targimag,axis=-1)),axis=-1)
        return reimpats, targpats
''' 
Returns the training generator.
'''
def dataloaderReturn(path_json):#simple dataloader take the path of the file folder  
    training_set = Dataset(path_json)# create a abj=dataset classs defain in the top it will call the constructor including the path return a list of path of all the data 
    training_generator = torch.utils.data.DataLoader(training_set, **params)
    """params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 8,
          'drop_last': True}"""

    return training_generator
