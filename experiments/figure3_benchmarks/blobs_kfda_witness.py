import warnings

from sklearn import model_selection
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use('ggplot')


import falkon
from testing_utils import snr_score


def sample_blobs(n, rows=3, cols=3, sep=1, rs=np.random):
    """Generate Blob-S for testing type-I error."""
    correlation = 0
    # generate within-blob variation
    mu = np.zeros(2)
    sigma = np.eye(2)
    X = rs.multivariate_normal(mu, sigma, size=n)
    corr_sigma = np.array([[1, correlation], [correlation, 1]])
    Y = rs.multivariate_normal(mu, corr_sigma, size=n)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=n) * sep
    X[:, 1] += rs.randint(cols, size=n) * sep
    Y[:, 0] += rs.randint(rows, size=n) * sep
    Y[:, 1] += rs.randint(cols, size=n) * sep
    return X, Y


def sample_blobs_Q(N1, sigma_mx_2, rows=3, cols=3, rs=None):
    """Generate Blob-D for testing type-II error (or test power)."""
    mu = np.zeros(2)
    sigma = np.eye(2) * 0.03
    rs = np.random
    X = rs.multivariate_normal(mu, sigma, size=N1)
    Y = rs.multivariate_normal(mu, np.eye(2), size=N1)
    # assign to blobs
    X[:, 0] += rs.randint(rows, size=N1)
    X[:, 1] += rs.randint(cols, size=N1)
    Y_row = rs.randint(rows, size=N1)
    Y_col = rs.randint(cols, size=N1)
    locs = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    for i in range(9):
        corr_sigma = sigma_mx_2[i]
        L = np.linalg.cholesky(corr_sigma)
        ind = np.expand_dims((Y_row == locs[i][0]) & (Y_col == locs[i][1]), 1)
        ind2 = np.concatenate((ind, ind), 1)
        Y = np.where(ind2, np.matmul(Y,L) + locs[i], Y)
    return X, Y

# blobs definitions used by Liu et al. 2020
sigma_mx_2_standard = np.array([[0.03, 0], [0, 0.03]])
sigma_mx_2 = np.zeros([9,2,2])
for i in range(9):
    sigma_mx_2[i] = sigma_mx_2_standard
    if i < 4:
        sigma_mx_2[i][0 ,1] = -0.02 - 0.002 * i
        sigma_mx_2[i][1, 0] = -0.02 - 0.002 * i
    if i==4:
        sigma_mx_2[i][0, 1] = 0.00
        sigma_mx_2[i][1, 0] = 0.00
    if i>4:
        sigma_mx_2[i][1, 0] = 0.02 + 0.002 * (i-5)
        sigma_mx_2[i][0, 1] = 0.02 + 0.002 * (i-5)

kernels = [falkon.kernels.GaussianKernel(bw) for bw in np.logspace(-3, 1, 10)]
penalties = np.logspace(-4, 3, 5)
parameter_grid = {'kernel': kernels, 'penalty': penalties}
folds = 2

n_list = [10, 20, 30, 40, 50] # number of samples in per mode
power_witness = []
power_mmd = []
np.random.seed(1)
for n in n_list:
    n_per_class = 9*n
    results_witness = []
    warnings.filterwarnings("ignore")
    pbar = tqdm(range(100))
    for i in pbar:
        s1,s2 = sample_blobs_Q(n_per_class, sigma_mx_2)
        s1test, s2test = sample_blobs_Q(n_per_class, sigma_mx_2)
        # s1,s2 = sample_blobs(n_per_class)
        # s1test, s2test = sample_blobs(n_per_class)

        X_train = np.concatenate((s1, s2))
        X_test = np.concatenate((s1test, s2test))
        Y_train = np.concatenate(([1 / n_per_class] * n_per_class, [-1 / n_per_class] * n_per_class))
        Y_test = np.concatenate(([1 / n_per_class] * n_per_class, [-1 / n_per_class] * n_per_class))
        # this is needed, since the cross-validation does not shuffle before creating the batches
        shuffle_train = np.random.permutation(2*n_per_class)
        shuffle_test = np.random.permutation(2*n_per_class)
        X_train, Y_train = X_train[shuffle_train], Y_train[shuffle_train]
        X_test, Y_test = X_test[shuffle_test], Y_test[shuffle_test]

        X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
        X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
        Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32).reshape(-1, 1)
        Y_test = torch.from_numpy(Y_test).to(dtype=torch.float32).reshape(-1, 1)

        # use all roughly all the data as nystrom centers
        M = (len(X_train) // folds) * (folds-1) // 1

        # define the estimator class
        estimator = falkon.FdaFalkon(kernel=falkon.kernels.GaussianKernel(1.), penalty=1e-3, M=M, maxiter=10,
                                  options=falkon.FalkonOptions(use_cpu=True))

        grid_search = model_selection.GridSearchCV(estimator, parameter_grid, cv=folds, scoring=snr_score)
        grid_search.fit(X_train, Y_train)
        k, reg = grid_search.best_params_['kernel'], grid_search.best_params_['penalty']
        # now run without Nystrom approximation
        M = len(X_train)
        kfda_witness = falkon.FdaFalkon(kernel=k, penalty=reg, M=M, maxiter=10,
                                        options=falkon.FalkonOptions(use_cpu=True))
        kfda_witness.fit(X_train, Y_train)
        # snr = snr_score(grid_search.best_estimator_, X_test, Y_test)
        # tau = np.sqrt(len(X_test)) * snr
        # p = 1 - norm.cdf(tau)
        # with permutations we return directly the pvalue
        p = snr_score(kfda_witness, X_test, Y_test, permutations=200)
        results_witness.append(1) if p < 0.05 else results_witness.append(0)

        pbar.set_description("n= %.0f" % n_per_class + " witness power: %.4f" % np.mean(results_witness))
    power_witness.append(np.mean(results_witness))

plt.plot(n_list, power_witness, label='kfda-witness')
plt.legend()
plt.xlabel("Samplesize")


# plt.show()
plt.savefig('type-II.pdf')
with open('type-II.npy', 'wb') as f:
    np.savez(f, n, power_witness)
