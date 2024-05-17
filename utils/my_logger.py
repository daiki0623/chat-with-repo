import logging

def set_logger(module_name):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=logging.INFO
    )
    logger = logging.getLogger(module_name)
    return logger