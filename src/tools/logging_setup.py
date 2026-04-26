import logging


def setup_bootstrap():
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s: %(message)s",
        level=logging.INFO,
    )
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    return logging.getLogger("bootstrap")
