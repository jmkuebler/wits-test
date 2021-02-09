import warnings

import numpy as np
import torch
from scipy.stats import norm
from tqdm import tqdm


import falkon
from testing_utils import snr_score, generate_data, mmd_U_p_value, kfda_mmd

# Fix Hyperparameters
kernel = falkon.kernels.GaussianKernel(0.2)
regularization = 1e-2 # adjust to reproduce Figure 6 of the appendix

# no. of folds for cross-validation
folds = 2

# samples per class
n_per_class = 100
ratios = [.1, .2, .3, .4, .5, .6, .7, .8, .9]

# list to store power
power_witness = []
power_mmd = []
power_mmd_opt = []
power_kfda_witness = []
power_kfda_boot = []

np.random.seed(1)
for ratio in ratios:
    # lists for individual outcomes
    results_witness = []
    results_mmd = []
    results_mmd_opt = []
    results_kfda_witness = []
    results_kfda_boot = []
    warnings.filterwarnings("ignore")   # to surpress warnings from the hacked Falkon implementation
    pbar = tqdm(range(1000))            # 100 iterations are also sufficient to check (and 10 times faster)
    for i in pbar:
        # generate data
        X_train, Y_train, X_test, Y_test = generate_data(dataset=('blobs', np.pi/4), n_per_class=n_per_class, ratio=ratio)
        X = torch.cat((X_train, X_test), dim=0)
        Y = torch.cat((Y_train, Y_test), dim=0)

        M = len(X_train)                # no Nystrom approximation, define witness via all points in the training set

        # MMD witness
        mmd_witness = falkon.Mmd(kernel=kernel, penalty=None, M=M, maxiter=10,
                                  options=falkon.FalkonOptions(use_cpu=True))
        mmd_witness.fit(X_train, Y_train)
        snr = snr_score(mmd_witness, X_test, Y_test)
        tau = np.sqrt(len(X_test)) * snr
        p = 1 - norm.cdf(tau)
        results_witness.append(1) if p < 0.05 else results_witness.append(0)

        # Compute MMD-boot test, only at first iteration (since we do not ues splitting)
        if ratio == ratios[0]:
            # nothing to select here, so we just play the interface.
            p_mmd = mmd_U_p_value([kernel], X, X, Y, Y, permutations=200)
            results_mmd.append(1) if p_mmd < 0.05 else results_mmd.append(0)

        # Compute opt-MMD test
        p_mmd = mmd_U_p_value([kernel], X_train, X_test, Y_train, Y_test, permutations=200)
        results_mmd_opt.append(1) if p_mmd < 0.05 else results_mmd_opt.append(0)

        # compute KFDA-witness with fixed regularization
        # define the estimator class
        kfda_witness = falkon.FdaFalkon(kernel=kernel, penalty=regularization, M=M, maxiter=10,
                                        options=falkon.FalkonOptions(use_cpu=True))
        kfda_witness.fit(X_train, Y_train)
        snr = snr_score(kfda_witness, X_test, Y_test)
        tau = np.sqrt(len(X_test)) * snr
        p = 1 - norm.cdf(tau)
        results_kfda_witness.append(1) if p < 0.05 else results_kfda_witness.append(0)

        if ratio == ratios[0]:
            # compute KFDA-boot only once
            p_kfda = float(kfda_mmd(kernel, X, Y, reg=regularization, permutations=200))
            results_kfda_boot.append(1) if p_kfda < 0.05 else results_kfda_boot.append(0)

        pbar.set_description("ratio= %.1f" % ratio + "kfda[0] power: %.4f" % np.mean(results_kfda_witness))
    power_kfda_witness.append(np.mean(results_kfda_witness))
    power_witness.append(np.mean(results_witness))
    power_mmd.append(np.mean(results_mmd))
    power_mmd_opt.append(np.mean(results_mmd_opt))
    power_kfda_boot.append(np.mean(results_kfda_boot))

with open('type-IIe-2.npy', 'wb') as f:
    np.savez(f, ratios, power_witness, power_mmd, power_mmd_opt, power_kfda_witness, power_kfda_boot)
