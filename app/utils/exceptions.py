from fastapi import HTTPException
from typing import Optional


class DedupException(Exception):
    """Base exception for deduplication service"""

    def __init__(self, message: str, code: str = "INTERNAL_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class AuthenticationError(DedupException):
    """Authentication related errors"""

    def __init__(self, message: str = "Invalid or missing authentication token"):
        super().__init__(message, "AUTHENTICATION_FAILED")


class CustomerNotFoundError(DedupException):
    """Customer not found errors"""

    def __init__(self, transaction_id: str):
        super().__init__(f"Customer {transaction_id} not found", "CUSTOMER_NOT_FOUND")


class InvalidRequestError(DedupException):
    """Invalid request errors"""

    def __init__(self, message: str):
        super().__init__(message, "INVALID_REQUEST")


class VectorServiceError(DedupException):
    """Vector service related errors"""

    def __init__(self, message: str):
        super().__init__(message, "VECTOR_SERVICE_ERROR")


def create_http_exception(exc: DedupException) -> HTTPException:
    """Convert DedupException to HTTPException"""
    status_code_map = {
        "AUTHENTICATION_FAILED": 401,
        "CUSTOMER_NOT_FOUND": 404,
        "INVALID_REQUEST": 400,
        "VECTOR_SERVICE_ERROR": 500,
        "INTERNAL_ERROR": 500
    }

    status_code = status_code_map.get(exc.code, 500)

    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": ""
            }
        }
    )