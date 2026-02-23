import os

USE_CACHE = os.getenv('TNT_USE_CACHE') == '1'


def default_cache_callback(*args):
    return USE_CACHE
