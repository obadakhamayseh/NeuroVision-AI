

from __future__ import annotations

import logging
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from backend.app.config.settings import settings
from backend.app.config.logging import configure_logging
from backend.app.middleware.logging import LoggingMiddleware
from backend.app.middleware.cors import setup_cors
from backend.app.exceptions import register_exception_handlers
from backend.app.api.v1.predict import router as predict_router
from backend.app.api.v1.health import router as health_router
from backend.app.api.v1.info import router as info_router
from backend.app.schemas.response import APIResponse

configure_logging()
logger = logging.getLogger("brain_tumor_api")

app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

from fastapi.staticfiles import StaticFiles
from pathlib import Path
results_path = Path("ml/artifacts/results")
results_path.mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=str(results_path)), name="results")

app.add_middleware(LoggingMiddleware)
setup_cors(app)

register_exception_handlers(app)

@app.get(
    "/",
    response_model=APIResponse,
    status_code=status.HTTP_200_OK,
    summary="Get API root metadata",
    description="Returns metadata about the NeuroVision AI brain tumor classification REST service."
)
async def get_root() -> APIResponse:
    
    return APIResponse(
        success=True,
        message="Welcome to the NeuroVision AI Brain MRI classification service API.",
        metadata={
            "api_name": settings.API_TITLE,
            "version": settings.API_VERSION,
            "documentation_swagger": "/docs",
            "documentation_redoc": "/redoc",
            "status": "online"
        }
    )

app.include_router(health_router, prefix="/health", tags=["Diagnostics"])
app.include_router(info_router, prefix="/model", tags=["Model Information"])
app.include_router(predict_router, prefix="/predict", tags=["Inference"])

app.include_router(health_router, prefix=f"{settings.API_V1_PREFIX}/health", tags=["Diagnostics"])
app.include_router(info_router, prefix=f"{settings.API_V1_PREFIX}/model", tags=["Model Information"])
app.include_router(predict_router, prefix=f"{settings.API_V1_PREFIX}/predict", tags=["Inference"])

@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("NeuroVision AI FastAPI Backend Service Starting Up")
    logger.info("API version: %s", settings.API_VERSION)
    logger.info("Swagger docs URL: http://%s:%d/docs", settings.HOST, settings.PORT)
    logger.info("=" * 60)
