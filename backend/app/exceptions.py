

from __future__ import annotations

import logging
from typing import Any
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from backend.app.schemas.response import APIResponse

logger = logging.getLogger("brain_tumor_api")

class APIException(Exception):
    
    def __init__(self, status_code: int, message: str, details: Any = None) -> None:
        self.status_code = status_code
        self.message = message
        self.details = details
        super().__init__(message)

def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
        logger.error("API Error occurred: %s (status %d)", exc.message, exc.status_code)
        response_data = APIResponse(
            success=False,
            message=exc.message,
            data=None,
            metadata={"details": exc.details} if exc.details else None
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=response_data.dict(exclude_none=True)
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("Request validation failed: %s", exc.errors())
        details = [{"field": ".".join(map(str, error["loc"])), "issue": error["msg"]} for error in exc.errors()]
        response_data = APIResponse(
            success=False,
            message="Unprocessable Entity - Request validation failed.",
            data=None,
            metadata={"validation_errors": details}
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=response_data.dict()
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unexpected server exception: %s", str(exc), exc_info=True)
        response_data = APIResponse(
            success=False,
            message="Internal Server Error - An unexpected error occurred.",
            data=None,
            metadata={"error_details": str(exc)}
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=response_data.dict()
        )
