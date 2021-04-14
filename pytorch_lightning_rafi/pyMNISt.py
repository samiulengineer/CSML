import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.optim as optim
train = datasets.MNIST('', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ]))
#train = true allow a folder , download= true anable of 

test = datasets.MNIST(  '', 
                        train       =   False,
                        download    =   True,
                        transform   =   transforms.Compose([transforms.ToTensor()])
                        )

trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=False)


class Net(Module):
    #our class will inharite the nn.module class 
    """class description : this class is for creating our nn """
    def __init__(self):
        super().__init__()
        #building nn 
        self.fc1 = nn.Linear(28*28, 64)# nn.linear take input all the image are 28*28 out put how many node will be 
        self.fc2 = nn.Linear(64, 64)# input will the out put of the fc1 
        self.fc3 = nn.Linear(64, 64)# input will the out put of the fc2
        self.fc4 = nn.Linear(64, 10)# the output of this neuron will provide the resuelt and our output is 0 to 9 = 10 

    def forward(self, x):
        #passing the value in the nn 
        #relue help the nn from data exploding or from getting 0
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)

net = Net()


loss_function = nn.CrossEntropyLoss()# for claculatin information lose 
optimizer = optim.Adam(net.parameters(), lr=0.0001)# net.perameters() takes all the values and find out what can be adjested 
#lr is the step the nn will take to find the optimum output result  it also optimize weight 


for epoch in range(10): # 3 full passes over the data
    for data in trainset:  # `data` is a batch of data
        X, y = data  # X is the batch of features, y is the batch of targets.
        # sets gradients to 0 before loss calc. You will do this likely every step.
        output = net(X.view(-1,784))  # pass in the reshaped batch (recall they are 28x28 atm)
        loss = F.nll_loss(output, y)  # calc and grab the loss value
        loss.backward()  # apply this loss backwards thru the network's parameters
        optimizer.step()  # attempt to optimize weights to account for loss/gradients
    print(loss)  # print loss. We hope loss (a measure of wrong-ness) declines! 