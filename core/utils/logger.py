import os
import sys

import logging

# from tensorboardX import SummaryWriter

def log_config(cfg):
    def get_print_attrs(cfg):
        attrs = dict(cfg.__dict__)
        for k in ['logger', 'env_fn', 'offline_data']:
            del attrs[k]
        return attrs
    attrs = get_print_attrs(cfg)
    for param, value in attrs.items():
        cfg.logger.info('{}: {}'.format(param, value))


class Logger:
    def __init__(self, config, log_dir):
        self.config = config
        log_file = os.path.join(log_dir, 'log')
        self._logger = logging.getLogger(__name__)
        file_handler = logging.FileHandler(log_file, mode='w')
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(formatter)
        self._logger.addHandler(file_handler)
        self._logger.setLevel(logging.DEBUG)

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)
