from sklearn import datasets, model_selection
import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt

import falkon
from testing_utils import snr_score


X, Y = datasets.make_circles(n_samples=1000, shuffle=False, noise=0.1, factor=.9)

# fig, ax = plt.subplots(figsize=(7, 7))
# ax.scatter(X[Y == 0,0], X[Y == 0,1], alpha=0.1, marker='.')
# _ = ax.scatter(X[Y == 1,0], X[Y == 1,1], alpha=0.1, marker='.')
# plt.show()

#--- prepare data ---- #
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
    X, Y, test_size=0.5, shuffle=True)
X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32).reshape(-1, 1)
Y_test = torch.from_numpy(Y_test).to(dtype=torch.float32).reshape(-1, 1)
Y_train[Y_train == 0] = -1
Y_test[Y_test == 0] = -1

# define kernel and regularization
kernel = flk_kernel = falkon.kernels.GaussianKernel(1.)
regularization = 1.
kfda_witness = falkon.FdaFalkon(kernel=kernel, penalty=regularization, M=len(X_train))

# ---- STAGE I - Train KFDA witness -----
kfda_witness.fit(X_train, Y_train)

# ---- STAGE II - Compute p-value -----
snr = snr_score(kfda_witness, X_test, Y_test)
tau = np.sqrt(len(X_test)) * snr
p = 1 - norm.cdf(tau)

print("p value = ", p)