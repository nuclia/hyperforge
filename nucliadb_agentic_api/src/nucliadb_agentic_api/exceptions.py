class Conflict(Exception):
    """Raised when a resource already exists and cannot be created again."""


class NotFound(Exception):
    """Raised when a resource is not found."""
