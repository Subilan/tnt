import logging

logging.basicConfig(
    level=logging.INFO,  # 设置最低显示级别
    format='[%(asctime)s] [%(levelname)s] >>> %(message)s'
)


def get_logger(name):
    return logging.getLogger(name)
