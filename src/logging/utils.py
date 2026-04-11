from loguru import logger


def info(msg: str) -> None:
    logger.info(msg)


def log_structure(obj: object) -> None:
    logger.info("{}", obj)
