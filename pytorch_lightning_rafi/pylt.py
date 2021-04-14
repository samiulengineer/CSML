import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer



class Lit(pl.LightningModule):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Lit, self).__init__()
        # Fully connected neural network with one hidden layer
        self.input_size =  784
        self.l1 = nn.Linear(28*28, 500)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(500, 10)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        
        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        
        tensorboard_logs = {'train_loss': loss}
        # use key 'log'
        return {"loss": loss, 'log': tensorboard_logs}

    # define what happens for testing here

    def train_dataloader(self):
        # MNIST dataset
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, transform=transforms.ToTensor(), download=True
        )
        # Data loader
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=100, num_workers=4, shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        test_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, transform=transforms.ToTensor(),download=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=100, num_workers=4, shuffle=True
        )
        return test_loader
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28 * 28)

        # Forward pass
        outputs = self(images)
                        
        loss = F.cross_entropy(outputs, labels)
        return {"val_loss": loss}
    
    def validation_epoch_end(self, outputs):
        # outputs = list of dictionaries
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        # use key 'log'
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.001)

if __name__ == '__main__':
    
    model = Lit(784, 500, 10)
    trainer = Trainer(max_epochs=3, fast_dev_run=False, gpus=1)
    trainer.fit(model)
    