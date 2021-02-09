import functools
import warnings

import torch
from scipy.linalg import lapack as scll
import numpy as np

from falkon.la_helpers import potrf
from falkon.options import FalkonOptions
from falkon.utils.helpers import choose_fn
from falkon.optim.conjgrad import projector


__all__ = ("check_init", "inplace_set_diag_th", "inplace_add_diag_th",
           "lauum_wrapper", "fda_lauum_wrapper", "potrf_wrapper")


def check_init(*none_check):
    def _checker(fun):
        @functools.wraps(fun)
        def wrapper(self, *args, **kwargs):
            is_init = True
            for el in none_check:
                if getattr(self, el, None) is None:
                    is_init = False
                    break
            if not is_init:
                raise RuntimeError(
                    "FALKON preconditioner is not initialized. Please run "
                    "`init` before any other method on the "
                    "preconditioner.")
            return fun(self, *args, **kwargs)
        return wrapper
    return _checker


def inplace_set_diag_th(A: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    A.diagonal().copy_(k)
    return A


def inplace_add_diag_th(A: torch.Tensor, k: float) -> torch.Tensor:
    # Assumes M is square (or wide also works).
    # Need to use .diagonal() as .diag() makes a copy
    A.diagonal().add_(k)
    return A


def lauum_wrapper(A: torch.Tensor, upper: bool, use_cuda: bool, opt: FalkonOptions) -> torch.Tensor:
    if use_cuda:
        from falkon.ooc_ops.ooc_lauum import gpu_lauum
        return gpu_lauum(A, upper=upper, write_opposite=True, overwrite=True, opt=opt)
    else:
        Anp = A.numpy()
        lauum = choose_fn(Anp.dtype, scll.dlauum, scll.slauum, "LAUUM")
        sol, info = lauum(Anp, lower=int(not upper), overwrite_c=1)
        if info != 0:
            raise RuntimeError(f"Lapack LAUUM failed with error code {info}.")
        return torch.from_numpy(sol)


def fda_lauum_wrapper(A: torch.Tensor, Y: torch.Tensor, upper: bool, use_cuda: bool, opt: FalkonOptions) -> torch.Tensor:
    '''
    Adopted from above for FDA.
    Y are the labels needed for the covariance
    '''
    warnings.warn("This function is not implemented in the proper FALKON style yet.")

    if use_cuda:
        from falkon.ooc_ops.ooc_lauum import gpu_lauum
        return gpu_lauum(A, upper=upper, write_opposite=True, overwrite=True, opt=opt)
    else:

        # # old
        # Anp = A.numpy()
        # lauum = choose_fn(Anp.dtype, scll.dlauum, scll.slauum, "LAUUM")
        # sol, info = lauum(Anp, lower=int(not upper), overwrite_c=1)
        # if info != 0:
        #     raise RuntimeError(f"Lapack LAUUM failed with error code {info}.")

        # fda
        T = torch.triu(A, diagonal=0)
        # sol = T @ T.T # this is the KRR solution
        proj_p, proj_q = projector(Y)
        sol = T - torch.matmul(torch.matmul(T, proj_p), torch.transpose(proj_p, 0, 1)) - \
                               torch.matmul(torch.matmul(T, proj_q), torch.transpose(proj_q, 0, 1))
        sol = sol @ sol.T
        sol = torch.tril(sol, diagonal=0) + torch.triu(T, diagonal=1)

        return sol


def potrf_wrapper(A: torch.Tensor, clean: bool, upper: bool, use_cuda: bool, opt: FalkonOptions) -> torch.Tensor:
    if use_cuda:
        from falkon.ooc_ops.ooc_potrf import gpu_cholesky
        return gpu_cholesky(A, upper=upper, clean=clean, overwrite=True, opt=opt)
    else:
        return potrf(A, upper=upper, clean=clean, overwrite=True, cuda=False)
