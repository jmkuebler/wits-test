import dataclasses
import time
import warnings
from typing import Union, Optional, Callable

import torch

import falkon
from falkon.models.model_utils import FalkonBase
from falkon.options import *
from falkon.utils import TicToc
from falkon.utils.devices import get_device_info

__all__ = ("Mmd",)


def get_min_cuda_preconditioner_size(dt, opt: FalkonOptions) -> int:
    if dt == torch.float32:
        return opt.min_cuda_pc_size_32
    else:
        return opt.min_cuda_pc_size_64


def get_min_cuda_mmv_size(dt, opt: FalkonOptions) -> int:
    if dt == torch.float32:
        return opt.min_cuda_iter_size_32
    else:
        return opt.min_cuda_iter_size_64

def mmd_weights(Y):
    # TODO: maybe there is a one-liner for this or we should put it somehwere else
    n_P = len([Y > 0])
    n_Q = len([Y < 0])
    weights = Y.clone().detach()
    weights[weights > 0] = 1 / n_P
    weights[weights < 0] = - 1 / n_Q
    return weights


class Mmd(FalkonBase):
    """MMD adoption

    Parameters
    -----------
    kernel
        Object representing the kernel function used for KRR.
    penalty : float
        Amount of regularization to apply to the problem.
        This parameter must be greater than 0.
    M : int
        The number of Nystrom centers to pick. `M` must be positive,
        and lower than the total number of training points. A larger
        `M` will typically lead to better accuracy but will use more
        computational resources.
    center_selection : str or falkon.center_selection.CenterSelector
        The center selection algorithm. Implemented is only 'uniform'
        selection which can choose each training sample with the same
        probability.
    maxiter : int
        The number of iterations to run the optimization for. Usually
        fewer than 20 iterations are necessary, however this is problem
        dependent.
    seed : int or None
        Random seed. Can be used to make results stable across runs.
        Randomness is present in the center selection algorithm, and in
        certain optimizers.
    error_fn : Callable or None
        A function with two arguments: targets and predictions, both :class:`torch.Tensor`
        objects which returns the error incurred for predicting 'predictions' instead of
        'targets'. This is used to display the evolution of the error during the iterations.
    error_every : int or None
        Evaluate the error (on training or validation data) every
        `error_every` iterations. If set to 1 then the error will be
        calculated at each iteration. If set to None, it will never be
        calculated.
    options : FalkonOptions
        Additional options used by the components of the Falkon solver. Individual options
        are documented in :mod:`falkon.options`.

    Examples
    --------
    Running Falkon on a random dataset

    >>> X = torch.randn(1000, 10)
    >>> Y = torch.randn(1000, 1)
    >>> kernel = falkon.kernels.GaussianKernel(3.0)
    >>> options = FalkonOptions(use_cpu=True)
    >>> model = Falkon(kernel=kernel, penalty=1e-6, M=500, options=options)
    >>> model.fit(X, Y)
    >>> preds = model.predict(X)

    References
    ----------
    .. [flk_1] Alessandro Rudi, Luigi Carratino, Lorenzo Rosasco, "FALKON: An optimal large
       scale kernel method," Advances in Neural Information Processing Systems 29, 2017.
    .. [flk_2] Giacomo Meanti, Luigi Carratino, Lorenzo Rosasco, Alessandro Rudi,
       "Kernel methods through the roof: handling billions of points efficiently,"
       arXiv:2006.10350, 2020.

    """

    def __init__(self,
                 kernel: falkon.kernels.Kernel,
                 penalty: float,
                 M: int,
                 center_selection: Union[str, falkon.center_selection.CenterSelector] = 'uniform',
                 maxiter: int = 20,
                 seed: Optional[int] = None,
                 error_fn: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
                 error_every: Optional[int] = 1,
                 options: Optional[FalkonOptions] = None,
                 ):
        super().__init__(kernel, M, center_selection, seed, error_fn, error_every, options)
        self.penalty = penalty
        self.maxiter = maxiter
        self._init_cuda()

    def fit(self,
            X: torch.Tensor,
            Y: torch.Tensor,
            Xts: Optional[torch.Tensor] = None,
            Yts: Optional[torch.Tensor] = None):
        """DEfines the MMD witness only on the Nystrom centers. There is no Falkon involved here! We usually use
        the full sample to get quadratic time MMD estimates!

        Parameters
        -----------
        X : torch.Tensor
            The tensor of training data, of shape [num_samples, num_dimensions].
            If X is in Fortran order (i.e. column-contiguous) then we can avoid
            an extra copy of the data.
        Y : torch.Tensor
            The tensor of training targets, of shape [num_samples, num_outputs].
            If X and Y represent a classification problem, Y can be encoded as a one-hot
            vector.
            If Y is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.
        Xts : torch.Tensor or None
            Tensor of validation data, of shape [num_test_samples, num_dimensions].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Xts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.
        Yts : torch.Tensor or None
            Tensor of validation targets, of shape [num_test_samples, num_outputs].
            If validation data is provided and `error_fn` was specified when
            creating the model, they will be used to print the validation error
            during the optimization iterations.
            If Yts is in Fortran order (i.e. column-contiguous) then we can avoid an
            extra copy of the data.

        Returns
        --------
        model: Falkon
            The fitted model
        """
        if self.M != len(X):
            warnings.warn("Setting M to max. So far we assume that the MMD witness is defined with all the data. But M is smaller "
                          "than the whole data. ")
            self.M = len(X)

        X, Y, Xts, Yts = self._check_fit_inputs(X, Y, Xts, Yts)
        # Todo: add check whether Y only has +-1 entries

        dtype = X.dtype

        # Decide whether to use CUDA for preconditioning based on M
        _use_cuda_preconditioner = (
                self.use_cuda_ and
                (not self.options.cpu_preconditioner) and
                self.M >= get_min_cuda_preconditioner_size(dtype, self.options)
        )
        _use_cuda_mmv = (
                self.use_cuda_ and
                X.shape[0] * X.shape[1] * self.M / self.num_gpus >= get_min_cuda_mmv_size(dtype, self.options)
        )

        self.fit_times_ = []
        self.ny_points_ = None
        self.alpha_ = None

        t_s = time.time()
        # noinspection PyTypeChecker
        # ny_points: Union[torch.Tensor, falkon.sparse.SparseTensor] = self.center_selection.select(X, None, self.M)
        # For the FDA preconditioner we need the labels
        ny_points: Union[torch.Tensor, falkon.sparse.SparseTensor] = self.center_selection.select(X, Y, self.M)

        if self.use_cuda_:
            ny_points = ny_points.pin_memory()

        self.alpha_ = mmd_weights(ny_points[1])
        self.ny_points_ = ny_points
        return self

    def _predict(self, X, ny_points, alpha: torch.Tensor) -> torch.Tensor:
        if ny_points is None:
            # Then X is the kernel itself
            return X @ alpha
        _use_cuda_mmv = (
                alpha.device.type == "cuda" or (
                self.use_cuda_ and
                X.shape[0] * X.shape[1] * self.M / self.num_gpus >= get_min_cuda_mmv_size(
            X.dtype, self.options)
        )
        )
        mmv_opt = dataclasses.replace(self.options, use_cpu=not _use_cuda_mmv)
        if type(ny_points) is tuple:
            # for fda ny_points contains the labels
            return self.kernel.mmv(X, ny_points[0], alpha, opt=mmv_opt)
        else:
            return self.kernel.mmv(X, ny_points, alpha, opt=mmv_opt)

    def to(self, device):
        self.alpha_ = self.alpha_.to(device)
        self.ny_points_ = self.ny_points_.to(device)
        return self
