import numpy as np
import matplotlib.pyplot as plt

# Read data
train = np.loadtxt('data/trainingIa.dat')
valid = np.loadtxt('data/validationIa.dat')
train_with_grad = np.loadtxt('data/trainingIb.dat')

x_train, y_train = train[:, 0], train[:, 1]
x_grad, y_grad, g_grad = train_with_grad[:, 0], train_with_grad[:, 1], train_with_grad[:, 2]
x_valid, y_valid = valid[:, 0], valid[:, 1]

# ----------Part I (a)----------
def fit(n, x_train, y_train, x_valid, y_valid):
    """
    :param n: order
    :param x_train: training data - x
    :param y_train: training data - V(x)
    :param x_valid: validation data - x
    :param y_valid: validation data - V(x)
    """
    V = np.vander(x_train, N=n, increasing=True)
    theta = np.linalg.inv(V.T @ V) @ V.T @ y_train
    train_mse = np.mean((V @ theta - y_train)**2)
    V_theta = np.vander(x_valid, N=n, increasing=True) @ theta
    valid_mse = np.mean((V_theta - y_valid)**2)
    return theta, valid_mse, train_mse

# check what number of degree n can reach the error < 1.0e-3
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

# plot the figure mse against degree n
plt.figure()
plt.yscale('log')
plt.plot(ns, mses, marker='o')
plt.axhline(y=1e-3, linestyle='--', color='red')
plt.xlabel('degree n')
plt.ylabel('MSE (validation)')
plt.title('MSE against n')
plt.legend()
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
    plt.plot(sizes, valid_mses, marker='o', label='valid mse')
    plt.plot(sizes, train_mses, label='train mse')
    plt.xlabel('training points')
    plt.ylabel('MSE')
    plt.title('MSE(valid and train) against the number of data m')
    plt.legend()
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
    plt.title('fitting polynomial')
    plt.legend()
    plt.show()

# ----------Part I (b)----------
def fit_with_grad(n, x_grad, y_grad, g_grad, x_valid, y_valid, lamb=1.0):
    """
    :param n: order
    :param x_grad: training data - x
    :param y_grad: training data - V(x)
    :param g_grad: training data - V'(x)
    :param x_valid: validation data - x
    :param y_valid: validation data - V(x)
    :param lamb: lambda
    """
    V = np.vander(x_grad, N=n, increasing=True)
    G = np.zeros_like(V, dtype=V.dtype)
    for j in range(1, n):
        G[:, j] = j * x_grad ** (j - 1)
    lhs = V.T @ V + lamb * G.T @ G
    rhs = V.T @ y_grad + lamb * G.T @ g_grad
    theta = np.linalg.inv(lhs) @ rhs
    train_mse = np.mean((V @ theta - y_grad) ** 2) + lamb * np.mean((G @ theta - g_grad) ** 2)
    V_theta = np.vander(x_valid, N=n, increasing=True) @ theta
    valid_mse = np.mean((V_theta - y_valid) ** 2)
    return theta, valid_mse, train_mse

# check what number of degree n can reach the error < 1.0e-3
ns = np.arange(1, 21)
mses_grad = []
thetas_grad = {}
n_optim_grad = None
least_valid_mse = float('inf')
for n in ns:
    theta_grad, valid_mse_grad, train_mse_grad = fit_with_grad(n, x_grad, y_grad, g_grad, x_valid, y_valid)
    if valid_mse_grad < least_valid_mse:
        n_optim_grad = n
        least_valid_mse = valid_mse_grad
    mses_grad.append(valid_mse_grad)
    thetas_grad[n] = theta_grad
print(f"n* = {n_optim_grad}, MSE = {mses_grad[n_optim_grad - 1] if n_optim_grad else None}")

# plot the figure mse against degree n
plt.figure()
plt.yscale('log')
plt.plot(ns, mses_grad, marker='o')
plt.axhline(y=1e-3, linestyle='--', color='red')
plt.xlabel('degree n')
plt.ylabel('MSE (validation)')
plt.title('MSE against n')
plt.legend()
plt.show()

if n_optim_grad is not None:
    theta_grad = thetas_grad[n_optim_grad]
    grid = np.linspace(x_grad.min(), x_grad.max(), 400)
    yhat_grad = np.vander(grid, n_optim_grad, increasing=True) @ theta_grad

    plt.figure(figsize=(12, 9))
    plt.scatter(x_grad, y_grad, label='train')
    plt.scatter(x_valid, y_valid, s=10, label='val')
    plt.plot(grid, yhat, label=f'fit (n={n_optim_grad})')
    plt.plot(grid, yhat, label=f'fit (n={n_optim}) without grad')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('V(x)')
    plt.title('fitting polynomial')
    plt.legend()
    plt.show()