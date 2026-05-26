import logging

logger = logging.getLogger("nucliadb_agentic_api")

SERVICE_NAME = "nucliadb_agentic_api"


# Define the filter
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return (
            record.args is not None
            and len(record.args) >= 3
            and record.args[2]  # type: ignore
            not in ("/", "/metrics", "/health/alive", "/health/ready")
        )


# Add filter to the logger
logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
