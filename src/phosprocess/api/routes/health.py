"""Health-check routes."""

from fastapi import APIRouter

from phosprocess.api.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check whether the API process is running",
)
def health() -> HealthResponse:
    """Return a lightweight process health check."""

    return HealthResponse()
