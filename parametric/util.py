import numpy as np
from mpmath import mp

mp.dps = 500

def intersect(itv1, itv2):
    itv = [max(itv1[0], itv2[0]), min(itv1[1], itv2[1])]
    if itv[0] > itv[1]:
        return None    
    return itv

def solve_linear_inequality(u, v): #u + vz < 0
    u = float(u)
    v = float(v)
    if (v > -1e-16 and v < 1e-16):
        if (u <= 0):
            return [-np.inf, np.inf]
        else:
            print('error', u, v)
            return None
    if (v < 0):
        return [-u/v, np.inf]
    return [-np.inf, -u/v]

def get_dnn_interval(X, u, v, model):
    layers = []

    for name, param in model.named_children():
        temp = dict(param._modules)
        
        for layer_name in temp.values():
            if ('Linear' in str(layer_name)):
                layers.append('Linear')
            elif ('ReLU' in str(layer_name)):
                layers.append('ReLU')

    ptr = 0
    itv = [-np.inf, np.inf]
    weight = None
    bias = None
    for name, param in model.named_parameters():
        if (layers[ptr] == 'Linear'):
            if ('weight' in name):
                weight = np.asarray(param.data.cpu())
            elif ('bias' in name):
                bias = np.asarray(param.data.cpu()).reshape(-1, 1)
                bias = bias.dot(np.ones((1, X.shape[0]))).T
                ptr += 1
                X = X.dot(weight.T) + bias
                u = u.dot(weight.T) + bias
                v = v.dot(weight.T)

        if (ptr < len(layers) and layers[ptr] == 'ReLU'):
            ptr += 1
            sub_itv = [-np.inf, np.inf]
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    if X[i][j] > 0:
                        sub_itv = intersect(sub_itv, solve_linear_inequality(-u[i][j], -v[i][j]))
                    else:
                        sub_itv = intersect(sub_itv, solve_linear_inequality(u[i][j], v[i][j]))
                        X[i][j] = 0
                        u[i][j] = 0
                        v[i][j] = 0
            itv = intersect(itv, sub_itv)
    # with open('count.txt', 'a') as f:
    #     f.write(f'{cnt}\n')
    return itv, u, v
def compute_u_v(x, eta, d):
    sq_norm = (np.linalg.norm(eta)) ** 2

    e1 = np.identity(d) - (np.dot(eta, eta.T)) / sq_norm
    u = np.dot(e1, x)

    v = eta / sq_norm

    return u, v


def construct_test_statistic(x, binary_vec, d):
    vector_1_S_a = np.zeros(d)
    vector_1_S_b = np.zeros(d)

    n_a = 0
    n_b = 0

    for i in range(d):
        if binary_vec[i] == 0:
            n_a = n_a + 1
            vector_1_S_a[i] = 1.0

        elif binary_vec[i] == 1:
            n_b = n_b + 1
            vector_1_S_b[i] = 1.0

    vector_1_S_a = np.reshape(vector_1_S_a, (vector_1_S_a.shape[0], 1))
    vector_1_S_b = np.reshape(vector_1_S_b, (vector_1_S_b.shape[0], 1))

    first_element = np.dot(vector_1_S_a.T, x)[0][0]
    second_element = np.dot(vector_1_S_b.T, x)[0][0]

    etaTx = first_element / n_a - second_element / n_b

    eta = vector_1_S_a / n_a - vector_1_S_b / n_b

    return eta, etaTx


def construct_z(binary_vec, list_zk, list_results):
    z_interval = []

    for i in range(len(list_results)):
        if np.array_equal(binary_vec, list_results[i]):
            z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

    new_z_interval = []

    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    z_interval = new_z_interval
    return z_interval


def pivot_with_specified_interval(z_interval, eta, etaTx, cov, tn_mu):
    tn_sigma = np.sqrt(np.dot(np.dot(eta.T, cov), eta))[0][0]
    # print(tn_sigma)
    numerator = 0
    denominator = 0

    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etaTx >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etaTx >= al) and (etaTx < ar):
            numerator = numerator + mp.ncdf((etaTx - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator)
    else:
        return None