import numpy as np
import matplotlib.pyplot as plt
from model import SimpleModel
import torch
import statsmodels.api as sm

import gen_data
import util
import parametric_si


def run():
    d = 8
    threshold = 20

    mu_1 = 0
    mu_2 = 1

    X, _ = gen_data.generate(1, d, mu_1, mu_2)

    hidden_dims = [2, 2]
    model = SimpleModel(input_dim=d, hidden_dims=hidden_dims)
    model.load_state_dict(torch.load('./weights/model.pth', weights_only=True))

    x_obs = X[0]
    # print("===========")
    # print('observe', x_obs)
    # print("===========")
    x_obs = x_obs.reshape((d, 1))

    prediction = model.forward(torch.tensor(x_obs.reshape(1, d), dtype=torch.float32)).detach().numpy()[0]

    binary_vec = []

    for each_e in prediction:
        if each_e <= 0.5:
            binary_vec.append(0)
        else:
            binary_vec.append(1)

    # print('observe', binary_vec)

    eta, etaTx = util.construct_test_statistic(x_obs, binary_vec, d)
    # print(etaTx)
    # print("\n")
    u, v = util.compute_u_v(x_obs, eta, d)

    list_zk, list_results = parametric_si.run_parametric_si(u, v, model, d, threshold)
    z_interval = util.construct_z(binary_vec, list_zk, list_results)

    # print(z_interval)

    cov = np.identity(d)

    mu_vec = np.array([mu_1, mu_1, mu_2, mu_2, mu_2, mu_2, mu_1, mu_1]).reshape((d, 1))
    tn_mu = np.dot(eta.T, mu_vec)[0][0]

    pivot = util.pivot_with_specified_interval(z_interval, eta, etaTx, cov, tn_mu)

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






