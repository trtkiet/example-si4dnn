from util import *
import torch
from multiprocessing import Pool
import os
from model import WDGRL
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kstest

def run_fpr(self):
    # np.random.seed(52907)
    _, Model = self
    # Create a new instance of the WDGRL model (same architecture as before)
    ns, nt, d = 150, 25, 1
    mu_s, mu_t = 0, 2
    delta_s, delta_t = [0, 1, 2, 3, 4], [0]
    xs, ys = gen_data(mu_s, delta_s, ns, d)
    xt, yt = gen_data(mu_t, delta_t, nt, d)

    Model.generator = Model.generator.cuda()

    xs = torch.FloatTensor(xs)
    ys = torch.LongTensor(ys)
    xt = torch.FloatTensor(xt)
    yt = torch.LongTensor(yt)

    xs_hat = Model.extract_feature(xs.cuda())
    xt_hat = Model.extract_feature(xt.cuda())
    x_hat = torch.cat([xs_hat, xt_hat], dim=0)

    xs_hat = xs_hat.cpu()
    xt_hat = xt_hat.cpu()
    x_hat = x_hat.cpu()
    xs = xs.cpu()
    xt = xt.cpu()
    ys = ys.cpu()
    yt = yt.cpu()
    O = max_sum(x_hat.numpy())
    if (O < ns):
        return run_fpr(self)
    else:
        O = [O - ns]   
    yt_hat = np.zeros((nt, 1))
    yt_hat[O] = 1
    Oc = list(np.where(yt_hat == 0)[0])
    X = np.vstack((xs, xt))
    X = torch.FloatTensor(X)
    j = np.random.choice(O)
    etj = np.zeros((nt, 1))
    etj[j][0] = 1
    etOc = np.zeros((nt, 1))
    etOc[Oc] = 1
    etaj = np.vstack((np.zeros((ns, 1)), etj-(1/len(Oc))*etOc))

    etajTx = etaj.T.dot(X)
    
    # print(f'Anomaly index: {O[0] + ns}')
    # print(f'etajTX: {etajTx}')
    mu = np.vstack((np.full((ns,1), mu_s), np.full((nt,1), mu_t)))
    sigma = np.identity(ns+nt)
    etajTmu = etaj.T.dot(mu)
    etajTsigmaetaj = etaj.T.dot(sigma).dot(etaj)
    b = sigma.dot(etaj).dot(np.linalg.inv(etajTsigmaetaj))
    a = (np.identity(ns+nt) - b.dot(etaj.T)).dot(X)
    threshold = 20
    list_zk, list_Oz = run_parametric_wdgrl(X, etaj, ns+nt, threshold, Model, ns)
    CDF = cdf(etajTmu[0][0], np.sqrt(etajTsigmaetaj[0][0]), list_zk, list_Oz, etajTx[0][0], [O[0] + ns])
    p_value = 2 * min(CDF, 1 - CDF)
    print(f'p-value: {p_value}')
    return p_value

if __name__ == '__main__':
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    d = 1
    generator_hidden_dims = [10, 10, 10, 10, 10]
    critic_hidden_dims = [4, 4, 2, 1]
    Model = WDGRL(input_dim=d, generator_hidden_dims=generator_hidden_dims, critic_hidden_dims=critic_hidden_dims)
    index = None
    with open("model/models.txt", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            words = line[:-1].split("/")
            if words[1] == str(generator_hidden_dims) and words[2] == str(critic_hidden_dims):
                index = i
                break
    if index is None:
        print("Model not found")
        exit()
    check_point = torch.load(f"model/wdgrl_{index}.pth", map_location=Model.device, weights_only=True)
    Model.generator.load_state_dict(check_point['generator_state_dict'])
    Model.critic.load_state_dict(check_point['critic_state_dict'])
    Model.generator = Model.generator.cpu()

    max_iter = 1000
    alpha = 0.05
    list_model = [Model for i in range(max_iter)]
    reject = 0
    detect = 0
    list_p_value = []
    pool = Pool(initializer=np.random.seed)
    list_result = pool.map(run_fpr, zip(range(max_iter), list_model))
    pool.close()
    pool.join()

    for p_value in list_result:
        if p_value is not None:
            detect += 1
            list_p_value.append(p_value)

            if (p_value < alpha):
                reject += 1
    with open(f"results/fpr_parametric.txt", "w") as f:
        f.write(str(reject/detect) + '\n')
        f.write(str(kstest(list_p_value, 'uniform')) + '\n')
    plt.hist(list_p_value)
    plt.savefig(f'results/fpr_parametric.png')
    plt.close()

