import numpy as np
import matplotlib.pyplot as plt

# Read data
train = np.loadtxt('data/trainingIa.dat')
valid = np.loadtxt('data/validationIa.dat')

x_train, y_train = train[:,0], train[:,1]
x_valid, y_valid = valid[:,0], valid[:,1]

def fit(n, x_train, y_train, x_valid, y_valid):
    """
    :param n: order
    :param x_train: training data - x
    :param y_train: training data - y
    :param x_valid: validation data - x
    :param y_valid: validation data - y
    """
    A = np.vander(x_train, N=n, increasing=True)
    theta = np.linalg.inv(A.T @ A) @ A.T @ y_train
    train_mse = np.mean((A @ theta - y_train)**2)
    V_theta = np.vander(x_valid, N=n, increasing=True) @ theta
    valid_mse = np.mean((V_theta - y_valid)**2)
    return theta, valid_mse, train_mse

ns = np.arange(1, 21)
mses = []
thetas = {}
n_optim = None
for n in ns:
    theta, valid_mse, train_mse = fit(n, x_train, y_train, x_valid, y_valid)
    if valid_mse < 1e-3:
        n_optim = n
    mses.append(valid_mse)
    thetas[n] = theta

print(f"n* = {n_optim}, MSE = {mses[n_optim - 1] if n_optim else None}")

# MSE–n
plt.figure()
plt.yscale('log')
plt.plot(ns, mses, marker='o')
plt.axhline(y=1e-3, linestyle='--', color='red')
plt.xlabel('degree n')
plt.ylabel('MSE (validation)')
plt.show()

if n_optim is not None:
    m_all = len(x_train)
    sizes = range(20, m_all)
    valid_mses = []
    train_mses = []
    rng = np.random.default_rng(42)
    for mprime in sizes:
        idx = rng.choice(m_all, size=mprime, replace=False)
        theta, valid_mse, train_mse = fit(n_optim, x_train[idx], y_train[idx], x_valid, y_valid)
        valid_mses.append(valid_mse)
        train_mses.append(train_mse)
    plt.figure()
    plt.yscale('log')
    plt.plot(sizes, valid_mses, marker='o')
    plt.plot(sizes, train_mses)
    plt.xlabel('# training points')
    plt.ylabel('MSE (validation)')
    plt.show()

    theta = thetas[n_optim]
    grid = np.linspace(x_train.min(), x_train.max(), 400)
    yhat = np.vander(grid, n_optim, increasing=True) @ theta

    plt.figure()
    plt.scatter(x_train, y_train, label='train')
    plt.scatter(x_valid, y_valid, s=10, label='val')
    plt.plot(grid, yhat, label=f'fit (n={n_optim})')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.show()
