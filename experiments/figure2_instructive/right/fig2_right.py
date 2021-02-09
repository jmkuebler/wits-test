import warnings

from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


import falkon
from testing_utils import snr_score, generate_data, mmd_U_p_value


kernels = [falkon.kernels.GaussianKernel(bw) for bw in np.logspace(-3, 1, 10)]
penalties = np.logspace(-4, 3, 5)
parameter_grid = {'kernel': kernels, 'penalty': penalties}
folds = 2

n = [100, 200, 300, 400]
power_witness = []
power_mmd = []
power_mmd_witness = []

for n_per_class in n:
    results_witness = []
    results_mmd = []
    results_mmd_witness = []
    warnings.filterwarnings("ignore")
    pbar = tqdm(range(1))
    for i in pbar:
        X_train, Y_train, X_test, Y_test = generate_data(dataset=('blobs', np.pi/4), n_per_class=n_per_class)
        M = (len(X_train) // folds) * (folds-1) // 1

        # define the estimator class
        estimator = falkon.FdaFalkon(kernel=None, penalty=None, M=M, maxiter=10,
                                  options=falkon.FalkonOptions(use_cpu=True))
        grid_search = model_selection.GridSearchCV(estimator, parameter_grid, cv=folds, scoring=snr_score)
        #
        grid_search.fit(X_train, Y_train)
        k, reg = grid_search.best_params_['kernel'], grid_search.best_params_['penalty']
        # now run without Nystrom approximation
        M = len(X_train)
        kfda_witness = falkon.FdaFalkon(kernel=k, penalty=reg, M=M, maxiter=10, options=falkon.FalkonOptions(use_cpu=True))
        kfda_witness.fit(X_train, Y_train)

        snr = snr_score(kfda_witness, X_test, Y_test)
        # print('snr', snr)
        tau = np.sqrt(len(X_test)) * snr
        p = 1 - norm.cdf(tau)
        results_witness.append(1) if p < 0.05 else results_witness.append(0)

        # Compute opt-MMD test
        p_mmd = mmd_U_p_value(kernels, X_train, X_test, Y_train, Y_test, permutations=1000)
        results_mmd.append(1) if p_mmd < 0.05 else results_mmd.append(0)

        # compute opt-mmd-witness
        # define the estimator class
        kernel_opt = mmd_U_p_value(kernels, X_train, X_train, Y_train, Y_train, permutations=1000, return_optimized_kernel=True)

        mmd_witness = falkon.Mmd(kernel=kernel_opt, penalty=None, M=M, maxiter=10,
                                 options=falkon.FalkonOptions(use_cpu=True))
        mmd_witness.fit(X_train, Y_train)
        snr = snr_score(mmd_witness, X_test, Y_test)
        # print('snr', snr)
        tau = np.sqrt(len(X_test)) * snr
        p = 1 - norm.cdf(tau)
        results_mmd_witness.append(1) if p < 0.05 else results_mmd_witness.append(0)
        pbar.set_description("n= %.0f" % n_per_class + " witness power: %.4f" % np.mean(results_witness) + " mmd power: %.4f" % np.mean(results_mmd)
                             + " mmd-witness power: %.4f" % np.mean(results_mmd_witness))

    power_witness.append(np.mean(results_witness))
    power_mmd.append(np.mean(results_mmd))
    power_mmd_witness.append(np.mean(results_mmd_witness))
with open('fig2_right.npy', 'wb') as f:
    np.savez(f, n, power_witness, power_mmd, power_mmd_witness)
