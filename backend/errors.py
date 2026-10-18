"""Domain errors; HTTP responses never expose exception messages or local paths."""


class BackendError(Exception):
    """Base class for application failures."""


class CatalogueError(BackendError):
    """The trusted local regulation dictionary is invalid."""


class InvalidAnalysisResultError(BackendError):
    """The analyzer produced an inconsistent or unsupported result."""
