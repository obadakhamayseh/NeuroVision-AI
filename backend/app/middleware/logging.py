

from __future__ import annotations

import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("brain_tumor_api")

class LoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        start_time = time.perf_counter()

        logger.info(
            "HTTP Request | method=%s path=%s client=%s",
            request.method,
            request.url.path,
            request.client.host if request.client else "unknown"
        )
        
        response = await call_next(request)
        
        process_time_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "HTTP Response | status_code=%d latency=%.2fms path=%s",
            response.status_code,
            process_time_ms,
            request.url.path
        )

        response.headers["X-Process-Time-MS"] = f"{process_time_ms:.2f}"
        
        return response
