import numpy as np
import matplotlib.pyplot as plt
from model import SimpleModel
import statsmodels.api as sm
import torch
import gen_data
import util


def run():
    d = 8
    X, _ = gen_data.generate(1, d, 2, 3)

    hidden_dims = [2, 2]
    model = SimpleModel(input_dim=d, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load('./weights/model.pth', weights_only=True))

    x = X[0]

    x = x.reshape((d, 1))

    # prediction = (util.sigmoid(np.dot(w1, x) + w0)).flatten()
    #
    # print(prediction)
    prediction = model.forward(torch.tensor(x.reshape(1, d), dtype=torch.float32)).detach().numpy()[0]
    # print(prediction)
    binary_vec = []

    for each_e in prediction:
        # print(each_e)
        if each_e <= 0.5:
            binary_vec.append(0)
        else:
            binary_vec.append(1)

    eta, etaTx = util.construct_test_statistic(x, binary_vec, d)
    u, v = util.compute_u_v(x, eta, d)

    Vminus = np.NINF
    Vplus = np.Inf

    itv = util.get_dnn_interval(x.reshape(1, d), u.reshape(1, d), v.reshape(1, d), model)
    Vminus = itv[0][0]
    Vplus = itv[0][1]

    cov = np.identity(d)

    mu_vec = np.array([2, 2, 3, 3, 3, 3, 2, 2]).reshape((d, 1))
    tn_mu = np.dot(eta.T, mu_vec)[0][0]

    pivot = util.pivot_with_specified_interval([[Vminus, Vplus]], eta, etaTx, cov, tn_mu)
    return pivot


if __name__ == '__main__':
    # np.random.seed(1)
    # run()

    max_iteration = 1200
    list_pivot = []

    for each_iter in range(max_iteration):
        print(each_iter)
        pivot = run()
        if pivot is not None:
            list_pivot.append(pivot)

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    # plt.switch_backend('agg')
    plt.plot(grid, sm.distributions.ECDF(np.array(list_pivot))(grid), 'r-', linewidth=6, label='Pivot')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('z_pivot.png', dpi=100)
    plt.show()






