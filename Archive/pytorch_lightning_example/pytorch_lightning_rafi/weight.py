import numpy as np 
import torch



# Weights and biases
w = torch.randn(2, 3, requires_grad=True)#2 row and 3 collom random values i means 2 vector with 3 values 
#requires_grad = True means we are allowing any altaratinon mainly in this part is derivative value 

b = torch.randn(2, requires_grad=True)#1 row of random values = 1 vector 

inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')



targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')


inputs = torch.from_numpy(inputs) #convert it to tensor 
targets = torch.from_numpy(targets) #convert it ot tensor 


#model to pradict 
def model(x):
    return x @ w.t() + b#w.t() make a transpos of the the metrix(more then 1 vector with n values )


preds = model(inputs)
print(preds)

""" the result will look like this after runnig the prediction 
what is the reason of this thing 
ans: we  picked the value of weight in rendom  and also the bias(main reason of it to prevent the network from being a death network)
tensor([[-36.5954,  83.9274],
        [-44.6611, 116.8989],
        [-55.7633,   4.9838],
        [-42.9439, 172.8393],
        [-37.0170,  72.9025]], grad_fn=<AddBackward0>)
[Finished in 0.8s]...
our target is to reach 
[56, 70], 
[81, 101], 
[119, 133], 
[22, 37], 
[103, 119]]
tunnig those value  to get to a target value one of the work of nn"""


#new concept information lose and gain 
#to find how good  is our prediction is 
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()#this find out the avg error in prediction numel() givees you the numbner of element in the vector


loss= mse(preds, targets )
print(loss)
"""tensor(15266.0498, grad_fn=<DivBackward0>)
[Finished in 0.7s]

"""
"""Adjust weights and biases to reduce the loss
The loss is a quadratic function of our weights and biases, 
and our objective is to find the set of weights where the loss is the lowest. 

If a gradient element is negative:

increasing the weight element's value slightly will decrease the loss
decreasing the weight element's value slightly will increase the loss

If a gradient element is positive:

increasing the weight element's value slightly will increase the loss
decreasing the weight element's value slightly will decrease the loss



new concept learning rate 
Learning rate gives the rate of speed where the gradient moves during gradient descent. """
""""def _backwords():
	preds = model(inputs)
	loss= mes(preds, targets)
	loss.backward()

def _grad_cal():

	with torch.no_grade():
		w -= w.grad*0.00001 #learnig rate 
        b -= b.grad*0.00001
        w.grad.zero_()
        b.grad.zero_()

for i in range(200):
	_backwords()
	_grad_cal()"""
for i in range(100):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()


preds = model(inputs)
loss = mse(preds, targets)
print(loss)
"""tensor([[  53.6454,    4.8120],
        [  72.6697,   25.6228],
        [ 117.5621,  106.9115],
        [  21.3113, -106.7704],
        [  87.5010,   94.3062]], grad_fn=<AddBackward0>)
tensor(3220.9219, grad_fn=<DivBackward0>)
tensor(386.6819, grad_fn=<DivBackward0>)
[Finished in 5.9s]"""