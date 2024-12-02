import os
from loguru import logger

LOGGING_DIR = os.getenv("LOGGING_DIR", "./logs")


def setup_logger():
    log_format = "<g>{time:YYYY-MM-DD HH:mm:ss.SSS}</g> | <lvl>{level:<8}</lvl> | <c>{process}</c> | <c>{module}</c>:<c>{function}</c> - <lvl>{message}</lvl>"

    logger.remove()
    logger.add(
        sink=os.sys.stdout,
        format=log_format,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        sink=LOGGING_DIR + "/risk-assessment-{time:YYYYMMDD}.log",
        format=log_format,
        rotation="00:00",
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )
