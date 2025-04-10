import tensorflow as tf
from tensorflow import keras
import torch
from model import SimpleModel

import numpy as np
import gen_data

def relu(x):
    return max(0, x)

d = 8

list_mu_1 = [0]
list_mu_2 = [1]


X_train = None
y_train = None

# X_train, y_train = gen_data.generate(1000, d, 2, 3)
# X_test, y_test = gen_data.generate(5, d, 2, 3)


for i in range(len(list_mu_1)):
    no_sample = 1000
    X, y = gen_data.generate(no_sample, d, list_mu_1[i], list_mu_2[i])
    if X_train is None:
        X_train = X
        y_train = y
    else:
        X_train = np.vstack((X_train, X))
        y_train = np.concatenate((y_train, y))

hidden_dims = [2, 2]
model = SimpleModel(input_dim=d, hidden_dims=hidden_dims)


model.train(X_train, y_train, epochs=10)

checkpoint = model.state_dict()

torch.save(checkpoint, "./weights/model.pth")
print("Model saved successfully!")

# for i in range(X_test.shape[0]):
#     X_i = X_test[i, :].reshape((d, 1))
#     prediction = (sigmoid(np.dot(w1, X_i) + w0)).flatten()
#     binary_vec = []
#
#     for each_e in prediction:
#         if each_e <= 0.5:
#             binary_vec.append(0)
#         else:
#             binary_vec.append(1)
#
#     print(binary_vec)

# print("===============")
#
# predictions = model.predict(X_test)
#
# for element in predictions:
#     list_element = list(element)
#     # print(list_element)
#
#     binary_vec = []
#
#     for each_e in list_element:
#         if each_e <= 0.5:
#             binary_vec.append(0)
#         else:
#             binary_vec.append(1)
#
#     print(binary_vec)


