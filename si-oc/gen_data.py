import numpy as np


def generate(n, d, mu_1, mu_2):
    X = []
    y = []

    for i in range(n):
        X_i = []
        y_i = []

        for j in range(d):
            if (j < d/4) or (j >= 3*d/4):
                val = np.random.normal(mu_1, 1)
                y_i.append(0)
            else:
                val = np.random.normal(mu_2, 1)
                y_i.append(1)

            X_i.append(val)

        X.append(X_i)
        y.append(y_i)

    return np.array(X), np.array(y)


if __name__ == '__main__':
    X, y = generate(10, 8, 2, 8)

    print(X)
    print(y)
    print(X.shape)
    print(y.shape)