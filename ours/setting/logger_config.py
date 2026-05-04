import logging


def _setup_logger():
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    log_logger = logging.getLogger()
    log_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    log_logger.handlers = [console_handler]

    return log_logger


logger = _setup_logger()
