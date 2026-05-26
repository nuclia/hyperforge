class RetrievalAgentError(Exception):
    """Base class for exceptions in Retrieval Agents."""


class NotFoundError(RetrievalAgentError):
    """Exception raised when a requested resource is not found."""


class DriverNotFoundError(NotFoundError):
    """Exception raised when the specified driver is not found."""


class ProtectedWorkflowError(RetrievalAgentError):
    """Exception raised when a protected workflow is modified."""


class ParseExportError(RetrievalAgentError):
    """Exception raised when there is an error parsing the export file."""


class ExportEncryptionError(RetrievalAgentError):
    """Exception raised when there is an error with encryption."""


class InvalidTargetAgentError(RetrievalAgentError):
    """Exception raised when the target agent is invalid for an operation."""
