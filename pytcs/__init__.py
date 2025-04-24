__all__ = ["ScopeFile", "__version__"]

from importlib.metadata import PackageNotFoundError, version

from .pytcs import ScopeFile

try:
    __version__ = version("pytcs")
except PackageNotFoundError:
    # package is not installed
    pass
finally:
    del version, PackageNotFoundError
