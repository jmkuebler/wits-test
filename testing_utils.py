import warnings

from sklearn import datasets, model_selection
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')



def snr_score(estimator, x_test, y_test, permutations=None):
    pred = estimator.predict(x_test).reshape(-1, )
    pred = np.array(pred)
    y_test = np.array(y_test).reshape(-1, )

    p_samp = pred[y_test > 0]
    q_samp = pred[y_test < 0]
    # print(len(p_samp), len(q_samp))
    c = len(p_samp) / (len(p_samp) + len(q_samp))
    signal = (np.mean(p_samp) - np.mean(q_samp))
    if permutations==None:
        if c == 1 or c==0:
            return -500
        noise = np.sqrt(1/c * np.var(p_samp) + 1/(1-c) * np.var(q_samp))
        if noise == 0:
            return - 500
        snr = signal / noise
        # check for nan
        if snr != snr:
            return -500
        else:
            return snr
    else:
        p = 0
        for i in range(permutations):
            np.random.shuffle(pred)
            p_samp = pred[y_test > 0]
            q_samp = pred[y_test < 0]
            signal_perm = np.mean(p_samp) - np.mean(q_samp)

            if signal <= float(signal_perm):
                p += float(1 / permutations)
        # print(signal, p)
        return p # this is the corresponding SNR

def blobs(theta, samplesize, spots=(3, 3), sigma=[0.1, 0.3]):
    rotMatrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    gaussians = np.array([np.random.normal(0, sigma[0], samplesize), np.random.normal(0, sigma[1], samplesize)])
    data = rotMatrix @ gaussians
    shifts = [np.random.randint(0, spots[0], samplesize), np.random.randint(0, spots[1], samplesize)]
    data = np.add(data, shifts)
    return np.transpose(data)


def generate_data(dataset=('blobs', np.pi/4), n_per_class=100, ratio=0.5):
    n_per_class = n_per_class
    # X = np.concatenate((np.random.uniform(-1.,.5, n_per_class), np.random.uniform(-.5,1., n_per_class)))
    # X = np.concatenate((np.random.normal(-1,.5, n_per_class//2), np.random.normal(1,.5, n_per_class//2), np.random.normal(0,1., n_per_class)))
    # X = np.concatenate((np.random.normal(0., .5, n_per_class), np.random.normal(0, 1., n_per_class)))
    if dataset[0] == 'blobs':
        theta = dataset[1]
        X = np.concatenate((blobs(0, n_per_class), blobs(theta, n_per_class)))
    if dataset[0] == 'disjoint_uniform':
        X = np.concatenate((np.random.uniform(-1., 0., n_per_class), np.random.uniform(0., 1., n_per_class)))
        X = np.array([[x, 0] for x in X])
    if dataset[0] == 'diff_means':
        X = np.concatenate((np.random.normal(1, 1, n_per_class), np.random.normal(dataset[1], 1., n_per_class)))
        X = np.array([[x, 0] for x in X])

    # X = np.array([[x, 0] for x in X]) # needed if data is 1-d
    Y = np.concatenate(([1/n_per_class]*n_per_class, [-1/n_per_class]*n_per_class))

    # make a 50/50 split for stage I and II
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=1-ratio, stratify=Y)
    # convert to torch for Falkon
    X_train = torch.from_numpy(X_train).to(dtype=torch.float32)
    X_test = torch.from_numpy(X_test).to(dtype=torch.float32)
    Y_train = torch.from_numpy(Y_train).to(dtype=torch.float32).reshape(-1, 1)
    Y_test = torch.from_numpy(Y_test).to(dtype=torch.float32).reshape(-1, 1)
    return X_train, Y_train, X_test, Y_test


def mmd_u_snr(kernel, X_train, Y_train):
    # compute the U statistic MMD SNR as test power criterion.
    """compute value of MMD and std of MMD using kernel matrix."""
    # sort the data
    Y_train = torch.reshape(Y_train, (-1,))
    X1 = X_train[Y_train > 0]
    n1 = len(X1)
    X2 = X_train[Y_train < 0]
    n2 = len(X2)
    if n1 != n2:
        warnings.warn('Samples differ in size. Truncating the larger')
        n1, n2 = min(n1,n2), min(n1,n2)
        X1, X2 = X1[:n1], X2[:n2]
    Kx = torch.zeros([n1,n1], dtype=X1.dtype)
    Kx = kernel(X1, X1, Kx)
    Kxy = torch.zeros([n1,n2], dtype=X1.dtype)
    Kxy = kernel(X1, X2, Kxy)
    Ky = torch.zeros([n2,n2], dtype=X1.dtype)
    Ky = kernel(X2, X2, Ky)

    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    nx = Kx.shape[0]
    ny = Ky.shape[0]
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny, hh.sum(1)/ny) / ny
    V2 = hh.sum() / nx / nx
    varEst = 4*(V1 - V2**2)
    if varEst == 0.0:
        return -5
    snr = mmd2 / np.sqrt(varEst)
    return snr

def mmd_U_p_value(kernels, X_train, X_test, Y_train, Y_test, permutations=100, return_optimized_kernel=False):
    """
    Pipeline to first select optimal kernel for U-stat MMD. Then compute p-value via bootstrap
    """
    # preproces data to have same samplesize.
    Y_train = torch.reshape(Y_train, (-1,))
    X1 = X_train[Y_train > 0]
    n1 = len(X1)
    X2 = X_train[Y_train < 0]
    n2 = len(X2)
    Y1, Y2 = Y_train[Y_train > 0], Y_train[Y_train < 0]
    if n1 != n2:
        warnings.warn('Samples differ in size. Truncating the larger')
        n1, n2 = min(n1,n2), min(n1,n2)
        X1, X2 = X1[:n1], X2[:n2]
        Y1, Y2 = Y1[:n1], Y2[:n2]
    X_train = torch.cat((X1, X2), dim=0)
    Y_train = torch.cat((Y1, Y2), dim=0)
    # preproces data to have same samplesize.
    Y_test = torch.reshape(Y_test, (-1,))
    X1 = X_test[Y_test > 0]
    n1 = len(X1)
    X2 = X_test[Y_test < 0]
    n2 = len(X2)
    Y1, Y2 = Y_test[Y_test > 0], Y_test[Y_test < 0]

    if n1 != n2:
        warnings.warn('Samples differ in size. Truncating the larger')
        n1, n2 = min(n1,n2), min(n1,n2)
        X1, X2 = X1[:n1], X2[:n2]
        Y1, Y2 = Y1[:n1], Y2[:n2]
    X_test = torch.cat((X1, X2), dim=0)
    Y_test = torch.cat((Y1, Y2), dim=0)

    # 1. Select optimal kernel
    snr_opt = -50
    kernel_opt = kernels[0]
    for kernel in kernels:
        snr = mmd_u_snr(kernel, X_train, Y_train)
        if snr > snr_opt:
            snr_opt = snr
            kernel_opt = kernel
    if return_optimized_kernel:
        return kernel_opt
    # print(kernel_opt.sigma)
    n_X = len(Y_test[Y_test > 0])
    n_Y = len(Y_test[Y_test < 0])

    # 2. Compute U-stat MMD and simulate null via permutations. --- taken from https://github.com/fengliu90/DK-for-TST
    Y_test = torch.reshape(Y_test, (-1,))
    # sort the data by classes
    X = torch.cat((X_test[Y_test > 0], X_test[Y_test < 0]), 0)
    Y = torch.cat((Y_test[Y_test > 0], Y_test[Y_test < 0]), 0)
    K = torch.zeros(size=[len(X_test), len(X_test)])
    K = kernel_opt(X, X, K)
    n = K.shape[0]
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest >= est).float().mean()
    # print(p_val, est)
    # return est.item(), p_val.item(), rest
    return p_val


def projector(Y):
    n_P = torch.sum(Y[Y == +1])
    n_Q = torch.sum(-1 * Y[Y == -1])
    proj_p, proj_q = Y.clone().detach(), Y.clone().detach()
    proj_p[proj_p == 1] = 1. / torch.sqrt(n_P)
    proj_p[proj_p == -1] = 0.
    proj_q[proj_q == -1] = 1. / torch.sqrt(n_Q)
    proj_q[proj_q == 1] = 0.
    return proj_p, proj_q


def kfda_mmd(kernel, X, Y, reg, permutations=100):
    # preproces data to have same samplesize.
    Y = torch.reshape(Y, (-1,))
    X1 = X[Y > 0]
    n1 = len(X1)
    X2 = X[Y < 0]
    n2 = len(X2)
    Y1, Y2 = Y[Y > 0], Y[Y < 0]
    if n1 != n2:
        warnings.warn('Samples differ in size. Truncating the larger')
        n1, n2 = min(n1,n2), min(n1,n2)
        X1, X2 = X1[:n1], X2[:n2]
        Y1, Y2 = Y1[:n1], Y2[:n2]
    X = torch.cat((X1, X2), dim=0)
    Y = torch.cat((Y1, Y2), dim=0)

    n_X = len(Y[Y > 0])
    n_Y = len(Y[Y < 0])

    # 2. Compute U-stat MMD and simulate null via permutations. --- taken from https://github.com/fengliu90/DK-for-TST
    Y = torch.reshape(Y, (-1,))
    # sort the data by classes
    X = torch.cat((X[Y > 0], X[Y < 0]), 0)
    Y = torch.cat((Y[Y > 0], Y[Y < 0]), 0)
    K = torch.zeros(size=[len(X), len(X)])
    # compute kernel matrix
    K = kernel(X, X, K)
    n = K.shape[0]
    w_X = 1 / (n//2)
    w_Y = -1 / (n//2)
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    ests = torch.zeros(permutations+1)
    for i in range(permutations + 1):
        proj_p, proj_q = projector(ws[i])
        proj_p = torch.reshape(proj_p, (-1, 1))
        proj_q = torch.reshape(proj_q, (-1,1))
        R = K - torch.matmul(proj_p, torch.matmul(torch.transpose(proj_p, 0, 1), K)) - \
            torch.matmul(proj_q, torch.matmul(torch.transpose(proj_q, 0, 1), K))
        cov = torch.mm(torch.transpose(R, 0,1), R)  + len(X) * reg * K
        # print('condition number:', np.linalg.cond(cov))
        alpha, _ = torch.solve(torch.mm(K, ws[i].reshape(-1,1)), cov)
        # print(alpha)
        ests[i] = torch.mm(ws[i].reshape(1,-1), torch.mm(K, alpha))
        # print('KFda vs mmd', ests[i]/(torch.linalg.norm(alpha) * torch.linalg.norm(ws[i])), torch.mm(ws[i].reshape(1,-1), torch.mm(K, ws[i].reshape(-1,1)))/(torch.linalg.norm(ws[i])**2))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest >= est).float().mean()
    # return est.item(), p_val.item(), rest
    return float(p_val)


def mmd_rkhs_optimized(kernel, X_train, X_test, Y_train, Y_test, regularization, permutations=100):
    # preproces data to have same samplesize.
    Y_train = torch.reshape(Y_train, (-1,))
    X1 = X_train[Y_train > 0]
    n1 = len(X1)
    X2 = X_train[Y_train < 0]
    n2 = len(X2)
    Y1, Y2 = Y_train[Y_train > 0], Y_train[Y_train < 0]
    if n1 != n2:
        warnings.warn('Samples differ in size. Truncating the larger')
        n1, n2 = min(n1, n2), min(n1, n2)
        X1, X2 = X1[:n1], X2[:n2]
        Y1, Y2 = Y1[:n1], Y2[:n2]
    X_train = torch.cat((X1, X2), dim=0)
    Y_train = torch.cat((Y1, Y2), dim=0)
    # preproces data to have same samplesize.
    Y_test = torch.reshape(Y_test, (-1,))
    X1 = X_test[Y_test > 0]
    n1 = len(X1)
    X2 = X_test[Y_test < 0]
    n2 = len(X2)
    Y1, Y2 = Y_test[Y_test > 0], Y_test[Y_test < 0]

    if n1 != n2:
        warnings.warn('Samples differ in size. Truncating the larger')
        n1, n2 = min(n1, n2), min(n1, n2)
        X1, X2 = X1[:n1], X2[:n2]
        Y1, Y2 = Y1[:n1], Y2[:n2]
    X_test = torch.cat((X1, X2), dim=0)
    Y_test = torch.cat((Y1, Y2), dim=0)

    Y_train = torch.reshape(Y_train, (-1,))
    # sort the data by classes
    X_train = torch.cat((X_train[Y_train > 0], X_train[Y_train < 0]), 0)
    Y_train = torch.cat((Y_train[Y_train > 0], Y_train[Y_train < 0]), 0)

    Y_test = torch.reshape(Y_test, (-1,))
    # sort the data by classes
    X_test = torch.cat((X_test[Y_test > 0], X_test[Y_test < 0]), 0)
    Y_test = torch.cat((Y_test[Y_test > 0], Y_test[Y_test < 0]), 0)


    K_tr = torch.zeros(size=[len(X_train), len(X_train)])
    K_tr = kernel(X_train, X_train, K_tr)
    n_tr = K_tr.shape[0]
    K_tr_te = torch.zeros(size=[len(X_train), len(X_test)])
    K_tr_te = kernel(X_train, X_test, K_tr_te)
    K_te = torch.zeros(size=[len(X_test), len(X_test)])
    K_te = kernel(X_test, X_test, K_te)
    n_te = K_te.shape[0]

    # invert the covariance operator via operator-inversion lemma
    proj_p, proj_q = projector(Y_train)
    N = torch.eye(n_tr) - n_tr//2 *torch.outer(proj_p,proj_p) - n_tr//2 *torch.outer(proj_q,proj_q)
    inverse = torch.inverse(regularization * torch.eye(n_tr) + 1/ n_tr * torch.einsum('ij, jk, kl -> il', N, K_tr, N))
    K_opt = K_te - 1 / (n_tr) * torch.einsum('ij, ik, kl, lm, mo -> jo', K_tr_te, N, inverse, N, K_tr_te)

    # compute MMD with optimized kernel
    w_X = 1
    w_Y = -1
    n_X = n_te //2
    n_Y = n_te // 2
    ws = torch.full((permutations + 1, n_te), w_Y, dtype=K_opt.dtype, device=K_opt.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n_te)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K_opt, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K_opt.take(Y_inds * n_te + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K_opt.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    # return est.item(), p_val.item(), rest
    return p_val
