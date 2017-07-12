from sys import version_info
if version_info[0] == 3:
    PY3 = True
    from importlib import reload
    from functools import reduce
elif version_info[0] == 2:
    PY3 = False
else:
    raise EnvironmentError("sys.version_info refers to a version of "
        "Python neither 2 nor 3. This is not permitted. "
        "sys.version_info = {}".format(version_info))

from sde_solve import *

__all__ = []

__version__ = (0, 1, 0)

import sde_solve

__modules = [sde_solve]
map(reload, __modules)

from sde_solve import *

__all__ = reduce(lambda a, b: a + b, map(lambda mod: mod.__all__, __modules)) + ['__version__']
