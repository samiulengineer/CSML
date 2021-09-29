import torch
import pytorch_lightning as pl

# from argparse import ArgumentParser # to give access to user on hyperparameter
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST # required to download the dataset
from torchvision import transforms



''' Hyperparameter '''

learning_rate = 1e-3
hidden_dim = 128
batch_size = 32




class LitClassifier(pl.LightningModule):
    def __init__(self, hidden_dim, learning_rate): # constructor of LitClassifier
                                                            # learning_rate will be used in gradient descent loss function
        super().__init__() # constructor of pl.LightningModule
        self.save_hyperparameters() # save hidden_dim & learning_rate & other hyperparameter
                                    # hyperparameter = variables of the network structure

        self.l1 = torch.nn.Linear(28 * 28, self.hparams.hidden_dim) # 1st input layer
        self.l2 = torch.nn.Linear(self.hparams.hidden_dim, 10) # output = 10 (0,1,2,3,4,5,6,7,8,9)

    def forward(self, x): # forward propagation
        x = x.view(x.size(0), -1) # pass the input x to the network: -1 is used for fix the column as the x.size(0)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch # x = input, y = actual output
        y_pred = self(x) # y_hat = y_predicted, by giving input x we will  get a output (y_pred) 
        loss = F.cross_entropy(y_pred, y) # this calculate the log_softmax and nll_loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y) #this calculate log_softmax and nll_loss
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    ''' dataset preparation '''

    def setup(self,stage):
        
        dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        # return mnist_test,mnist_train,mnist_val


    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size)
    
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--hidden_dim', type=int, default=128)
    #     parser.add_argument('--learning_rate', type=float, default=0.0001)
    #     return parser



def main():

    # pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', default=32, type=int)
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    # args = parser.parse_args()

    # ------------
    # model
    # ------------
    model = LitClassifier(hidden_dim, learning_rate)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(max_epochs=3, fast_dev_run=False, gpus=4)
    trainer.fit(model)

    # ------------
    # testing
    # ------------
    trainer.test(model)


if __name__ == '__main__':
    main()
