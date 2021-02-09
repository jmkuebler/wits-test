import warnings

from sklearn import model_selection
import numpy as np
import pickle
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm


import falkon
from testing_utils import snr_score

# Load data
data = pickle.load(open('./HIGGS_TST.pckl', 'rb'))
dataX = data[0]
dataY = data[1]
# REPLACE above two lines with
# dataX = data[0]
# dataY = data[0]
# or
# dataX = data[1]
# dataY = data[1]
# for validating type-I error
del data

kernels = [falkon.kernels.GaussianKernel(bw) for bw in np.logspace(-3, 1, 10)]
penalties = np.logspace(-4, 3, 5)
parameter_grid = {'kernel': kernels, 'penalty': penalties}
folds = 2

n_list = [1000, 2000, 3000, 5000]
power_witness = []
np.random.seed(0)
for n in n_list:
    results_witness = []
    warnings.filterwarnings("ignore")
    pbar = tqdm(range(100))
    for i in pbar:
        ## ---- Draw Data ---- ###
        # Generate Higgs (P,Q)
        N1_T = dataX.shape[0]
        N2_T = dataY.shape[0]
        ind1 = np.random.choice(N1_T, n, replace=False)
        ind2 = np.random.choice(N2_T, n, replace=False)
        s1 = dataX[ind1,:4]
        s2 = dataY[ind2,:4]
        ind1 = np.random.choice(N1_T, n, replace=False)
        ind2 = np.random.choice(N2_T, n, replace=False)
        s1test = dataX[ind1,:4]
        s2test = dataY[ind2,:4]

        X_train = np.concatenate((s1, s2))
        X_test = np.concatenate((s1test, s2test))
        Y_train = np.concatenate(([1 / n] * n, [-1 / n] * n))
        Y_test = np.concatenate(([1 / n] * n, [-1 / n] * n))
        shuffle_train = np.random.permutation(2 * n)
        shuffle_test = np.random.permutation(2 * n)
        X_train, Y_train = X_train[shuffle_train], Y_train[shuffle_train]
        X_test, Y_test = X_test[shuffle_test], Y_test[shuffle_test]

        X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
        X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
        Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32).reshape(-1, 1)
        Y_test = torch.from_numpy(Y_test).to(dtype=torch.float32).reshape(-1, 1)
        # this is needed, since the cross-validation does not shuffle before creating the batches
        shuffle_train = np.random.permutation(2*n)
        shuffle_test = np.random.permutation(2*n)
        X_train, Y_train = X_train[shuffle_train], Y_train[shuffle_train]
        X_test, Y_test = X_test[shuffle_test], Y_test[shuffle_test]

        M = 500         # Use Falkon approximation, since Samplesizes are large
        # define the estimator class
        estimator = falkon.FdaFalkon(kernel=None, penalty=None, M=M, maxiter=10,
                                     options=falkon.FalkonOptions(use_cpu=True))

        # --- Stage I - train kfda witness with cross validation --- ##
        grid_search = model_selection.GridSearchCV(estimator, parameter_grid, cv=folds, scoring=snr_score)
        grid_search.fit(X_train, Y_train)
        k, reg = grid_search.best_params_['kernel'], grid_search.best_params_['penalty']
        kfda_witness = falkon.FdaFalkon(kernel=k, penalty=reg, M=M, maxiter=10,
                                        options=falkon.FalkonOptions(use_cpu=True))
        kfda_witness.fit(X_train, Y_train)

        # --- Stage II --- #
        p = snr_score(kfda_witness, X_test, Y_test, permutations=200)
        results_witness.append(1) if p < 0.05 else results_witness.append(0)

        pbar.set_description("n= %.0f" % n + " witness power: %.4f" % np.mean(results_witness))
    power_witness.append(np.mean(results_witness))

with open('higgs_kfda_witness.npy', 'wb') as f:
    np.savez(f, n, power_witness)
