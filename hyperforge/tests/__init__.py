import logging


def configure_logging():
    print("configuring tests logging")

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("httpcore.connection").setLevel(logging.ERROR)
    logging.getLogger("httpcore.http11").setLevel(logging.ERROR)
    logging.getLogger("asyncio").setLevel(logging.INFO)

    # Configure arag.memory logger
    arag_logger = logging.getLogger("arag.memory")
    arag_logger.setLevel(logging.DEBUG)
    arag_logger.propagate = True  # Ensures it bubbles up to root_logger


configure_logging()
