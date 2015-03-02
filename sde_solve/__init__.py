from sde_solve import *

__all__ = ['sde_platen_15']

__version__ = (0, 1, 0)

import sde_solve

__modules = [sde_solve]
map(reload, __modules)

from sde_solve import *

__all__ = reduce(lambda a, b: a+b, map(lambda mod: mod.__all__, __modules)) + ['__version__']
