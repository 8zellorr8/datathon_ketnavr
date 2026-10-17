"""Controlled errors with the frontend's stable JSON envelope."""
from .errors import BackendError


class APIError(BackendError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
