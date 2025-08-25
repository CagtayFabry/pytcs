__all__ = [
    "PytcsDeprecationWarning",
    "PytcsException",
    "PytcsWarning",
]


class PytcsWarning(Warning):
    """
    The base warning class to inherit for all pytcs warnings.
    """


class PytcsException(Exception):
    """
    The base exception class to inherit for all pytcs exceptions.
    """


class PytcsDeprecationWarning(PytcsWarning, DeprecationWarning):
    """
    A warning class to indicate a deprecated feature.
    """
