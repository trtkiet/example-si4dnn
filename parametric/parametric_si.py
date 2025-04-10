import numpy as np
import util
import torch

def run_parametric_si(u, v, model, d, threshold):
    zk = -threshold

    list_zk = [zk]
    list_results = []

    while zk < threshold:
        x = u + v*zk
        # print(x.flatten())

        prediction = model.forward(torch.tensor(x.reshape(1, d), dtype=torch.float32)).detach().numpy()[0]
        # print(prediction)
        binary_vec = []

        for each_e in prediction:
            # print(each_e)
            if each_e <= 0.5:
                binary_vec.append(0)
            else:
                binary_vec.append(1)

        # print(binary_vec)
        Vminus = np.NINF
        Vplus = np.Inf

        itv = util.get_dnn_interval(x.reshape(1, d), u.reshape(1, d), v.reshape(1, d), model)
        Vminus = itv[0][0]
        Vplus = itv[0][1]

        zk = Vplus + 0.0001
        # print(zk)
        # print("================")
        if zk < threshold:
            list_zk.append(zk)
        else:
            list_zk.append(threshold)

        list_results.append(binary_vec)

    return list_zk, list_results