# A Witness Two-Sample Test
All the utility functions we implemented are in the file `testing_utils.py`.

The provided implementation builts on two other code-bases:
1. We provide an implementation to estimate KFDA witnesses. To do so, we extend the FALKON (Rudi et al. (2017), Meanti et al. (2020)) code ([https://github.com/FalkonML/falkon](https://github.com/FalkonML/falkon)) 
with a method for KFDA - implemented via the new class `FdaFalkon`. We discuss this in Appendix C of our paper.
   
2. We provide benchmark experiments for deep optimized kernels. Therefore we reuse the experiments of Liu et al.(2020) 
   ([https://github.com/fengliu90/DK-for-TST](https://github.com/fengliu90/DK-for-TST)) and extend them with the proposed Witness approaches.

## Installing
The installation is tested for Python version 3.6. 
0. create virtualenvironment
1. `pip install -r requirements.txt`
2. `cd kfda_falkon`
3. `pip install -e .` (to install Falkon modules)

## Reproduce Experimental Results
Navigate to the experiment directory and follow the instructions in the respective `Readme.md`.

## Minimal Working Example `mwe.py` - Estimating p-values
```python
from sklearn import datasets, model_selection
import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt

import falkon
from testing_utils import snr_score


X, Y = datasets.make_circles(n_samples=1000, shuffle=False, noise=0.1, factor=.9)
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
```

## References
1. F. Liu, W. Xu, J. Lu, G. Zhang, A. Gretton, and D. J. Sutherland.   *Learning  deep  kernels  for  non-parametric  two-sample tests*. ICML, 2020.
2. A. Rudi, L. Carratino, and L. Rosasco. Falkon: *An optimal large scale kernel method*. NeurIPS, 2017.
3. G. Meanti, L. Carratino, L. Rosasco, and A. Rudi. *Kernel methods through the roof:  Handling billions of points efficiently*. NeurIPS, 2020.
