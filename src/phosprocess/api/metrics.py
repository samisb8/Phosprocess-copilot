"""Prometheus metrics for the PhosProcess API."""

from prometheus_client import Counter, Histogram

HTTP_REQUESTS_TOTAL = Counter(
    "phosprocess_http_requests_total",
    "Total number of HTTP requests.",
    ("method", "route", "status"),
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "phosprocess_http_request_duration_seconds",
    "HTTP request duration in seconds.",
    ("method", "route"),
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        20.0,
        30.0,
        60.0,
        120.0,
    ),
)
