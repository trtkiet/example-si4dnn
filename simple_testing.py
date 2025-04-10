import numpy as np
from model import SimpleModel
import gen_data
import torch


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


d = 8
hidden_dims = [2, 2]
model = SimpleModel(input_dim=d, hidden_dims=hidden_dims)
model.load_state_dict(torch.load('./weights/model.pth', weights_only=True))

mu_1 = 1
mu_2 = 10

X, _ = gen_data.generate(1, d, mu_1, mu_2)

x_obs = X[0]
x_obs = x_obs.reshape((1, d))

prediction = model.forward(torch.tensor(x_obs, dtype=torch.float32)).detach().numpy()[0]
print(prediction)
binary_vec = []

for each_e in prediction:
    if each_e <= 0.5:
        binary_vec.append(0)
    else:
        binary_vec.append(1)

print(binary_vec)
