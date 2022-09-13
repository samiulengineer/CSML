import numpy as np
import torch
import matplotlib.pyplot as plt


y_axis_data = []
for i in range(0, 100):
    y_axis_data.append(i)

x2 = torch.from_numpy(
    np.random.uniform(-10, 10, (100, 1)).astype(np.float32))
x1 = torch.from_numpy(
    np.random.uniform(-10, 10, (100, 1)).astype(np.float32))
a = torch.from_numpy(
    np.random.uniform(-100, 100, (100, 1)).astype(np.float32))
b = torch.from_numpy(
    np.random.uniform(-100, 100, (100, 1)).astype(np.float32))

y = np.zeros([100, 1])   
for i in range(100):
    y[i] = (a[i] * x1[i]) + (b[i] * x2[i])
    y=np.float32(y)
    
y = torch.Tensor(y)  #eited 

fig = plt.figure(figsize=(15, 30))
plt.plot(x1, y_axis_data, label="x1", linewidth=10)
plt.plot(x2, y_axis_data, label="x2", linewidth=10)
# plt.plot(a, y_axis_data, label="a", linewidth=10)
# plt.plot(b, y_axis_data, label="b", linewidth=10)
# plt.plot(y, y_axis_data, label="y", linewidth=10,
#          color="red")


plt.legend(prop={'size': 40})
plt.xticks(fontsize=36)
plt.yticks(fontsize=36)
plt.tight_layout(pad=0)
plt.savefig('lightning_logs/dataset_graph_x1_x2.jpg')
plt.show()
