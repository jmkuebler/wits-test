import os

from . import kernels, optim, preconditioner, center_selection, sparse
from .options import FalkonOptions
from .models import Falkon, LogisticFalkon, InCoreFalkon, FdaFalkon, Mmd

__all__ = ('Falkon', 'LogisticFalkon', 'InCoreFalkon', 'FdaFalkon', 'Mmd',
           'kernels', 'optim', 'preconditioner', 'center_selection', 'sparse',
           'FalkonOptions')


# Set __version__ attribute on the package
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'VERSION')) as version_file:
    __version__ = version_file.read().strip()

