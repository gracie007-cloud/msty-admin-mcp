"""
Msty Admin MCP - Error Handling

Standardized error codes and response helpers for consistent API responses.
"""

import json
from datetime import datetime
from typing import Optional


class ErrorCode:
    """Standardized error codes for consistent API responses"""
    DATABASE_NOT_FOUND = "DATABASE_NOT_FOUND"
    DATABASE_ERROR = "DATABASE_ERROR"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def make_error_response(
    error_code: str,
    message: str,
    suggestion: Optional[str] = None,
    details: Optional[dict] = None
) -> dict:
    """
    Create a standardized error response dict.

    Args:
        error_code: One of ErrorCode constants
        message: Human-readable error message
        suggestion: Optional suggestion for how to fix the issue
        details: Optional additional details about the error

    Returns:
        Standardized error response dict
    """
    response = {
        "success": False,
        "error": {
            "code": error_code,
            "message": message
        },
        "timestamp": datetime.now().isoformat()
    }
    if suggestion:
        response["error"]["suggestion"] = suggestion
    if details:
        response["error"]["details"] = details
    return response


def error_response(
    error_code: str,
    message: str,
    suggestion: Optional[str] = None,
    details: Optional[dict] = None
) -> str:
    """
    Create a standardized JSON error response string.

    Args:
        error_code: One of ErrorCode constants
        message: Human-readable error message
        suggestion: Optional suggestion for how to fix the issue
        details: Optional additional details about the error

    Returns:
        JSON string with standardized error format
    """
    return json.dumps(make_error_response(error_code, message, suggestion, details), indent=2)


def make_success_response(data: dict, message: Optional[str] = None) -> dict:
    """
    Create a standardized success response dict.

    Args:
        data: The response data
        message: Optional success message

    Returns:
        Standardized success response dict
    """
    response = {
        "success": True,
        "timestamp": datetime.now().isoformat(),
        **data
    }
    if message:
        response["message"] = message
    return response


def success_response(data: dict, message: Optional[str] = None) -> str:
    """
    Create a standardized JSON success response string.

    Args:
        data: The response data
        message: Optional success message

    Returns:
        JSON string with standardized success format
    """
    return json.dumps(make_success_response(data, message), indent=2, default=str)


__all__ = [
    "ErrorCode",
    "make_error_response",
    "error_response",
    "make_success_response",
    "success_response"
]
