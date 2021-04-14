import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision as tv
from PIL import Image


# Download the Dataset from terminal

# !wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
# !tar xvzf BSDS300-images.tgz
# !rm BSDS300-images.tgz


class NoisyDataset(Dataset):
  
  def __init__(self, in_path, mode='train', img_size=(180, 180), sigma=30):
    super(NoisyDataset, self).__init__()

    self.mode = mode #train or test
    self.in_path = in_path # ./BSDS300/images
    self.img_size = img_size # (180, 180)


    self.img_dir = os.path.join(in_path, mode)
    self.imgs = os.listdir(self.img_dir)
    self.sigma = sigma

  def __len__(self):
      return len(self.imgs)
  
  def __repr__(self):
      return "Dataset Parameters: mode={}, img_size={}, sigma={}".format(self.mode, self.img_size, self.sigma)
    
  def __getitem__(self, idx):

      img_path = os.path.join(self.img_dir, self.imgs[idx])
      clean_img = Image.open(img_path).convert('RGB')
      left = np.random.randint(clean_img.size[0] - self.img_size[0])
      top = np.random.randint(clean_img.size[1] - self.img_size[1])
      # .crop(left, upper, right, lower)
      cropped_clean = clean_img.crop([left, top, left+self.img_size[0], top+self.img_size[1]])
      transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  
      
      ground_truth = transform(cropped_clean)

      noisy = ground_truth + 2 / 255 * self.sigma * torch.randn(ground_truth.shape)
      
      # return {"noisy":noisy, "ground_truth":ground_truth} 
      return noisy,ground_truth



      


if __name__ == '__main__':
  
  test = NoisyDataset("/home/mdsamiul/InSAR-Coding/data/BSDS300/images/", mode="test")
  
  train_dataloader = DataLoader(test, batch_size=20, num_workers=4)

  ''' Output & Visualize the training data '''
    
    
  # print('train_dataset length {}'.format(len(test)))
  # print('type of train_dataset {}'.format(type(train_dataloader)))

  
  for batch_idx, batch in enumerate(train_dataloader):
      x,y= batch
      print(batch_idx)
      print(x.shape)
      print(y.shape)
      # print('Batch Index \t = {}'.format(batch_idx))
      # print('Input Shape \t = {}'.format(batch['noisy'].shape))
      # print('Coh Shape \t = {}'.format(batch['ground_truth'].shape))
      # print('mr Shape \t = {}'.format(batch['mr'].shape))
      # print('he Shape \t = {}'.format(batch['he'].shape))
      # print('ddays Shape \t = {}'.format(batch['ddays'].shape))
      # print('bperps Shape \t = {}'.format(batch['bperps'].shape))
      # print('conv1 \t = {}'.format(batch['conv1'].shape))
      # print('conv2 \t = {}'.format(batch['conv2'].shape))
      # print('Wrap recon phase = {}'.format(batch['wrap_recon_phase'].shape))


      break

