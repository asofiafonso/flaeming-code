import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from typing import Any


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors"""

    grey: str = "\x1b[38;20m"
    purple: str = "\x1b[35;20m"
    green: str = "\x1b[32;20m"
    yellow: str = "\x1b[33;20m"
    red: str = "\x1b[31;20m"
    bold_red: str = "\x1b[31;1m"
    reset: str = "\x1b[0m"
    text_format: str = "[%(levelname)s] - %(asctime)s - %(filename)s - %(name)s - %(funcName)s - %(lineno)d: %(message)s"

    FORMATS = {
        logging.DEBUG: purple + text_format + reset,
        logging.INFO: green + text_format + reset,
        logging.WARNING: yellow + text_format + reset,
        logging.ERROR: red + text_format + reset,
        logging.CRITICAL: bold_red + text_format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


configurations: dict[str, Any] = {
    "name": "flaeming",
    "format": "[%(levelname)s] - %(asctime)s - %(name)s - %(funcName)s - %(lineno)d : %(message)s",
    # "format": CustomFormatter(),
    "dir": ["log"],
    "full_path": False,
    "to_file": False,
    "level": "info",
    "existent_log_levels": {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    },
    "path_splitter": "/|\\\\",
    "rotate_unit": "m",
    "rotate_time": 60,
    "backup_count": 5,
    "enable": True,
}


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class Singleton(type):
    _instances: dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ConsoleHandlerFactory(metaclass=Singleton):
    """
    Class to create and ensure that only one instance of ConsoleHandler exists
    """

    def __init__(self):
        self.handler = None

    def get_handler(self, conf):
        if self.handler is not None:
            return self.handler

        console_handler = logging.StreamHandler(sys.stdout)
        if isinstance(conf["format"], str):
            console_handler.setFormatter(logging.Formatter(conf["format"]))
        else:
            console_handler.setFormatter(conf["format"])
        console_handler.setLevel(configurations["existent_log_levels"][conf["level"]])
        self.handler = console_handler
        return self.handler


class TimedRotatingFileHandlerFactory(metaclass=Singleton):
    """
    Class to create and ensure that only one instance of TimedRotatingFileHandler exists.
    """

    def __init__(self):
        self.handler = None

    def get_handler(self, conf):
        """
        Returns a TimedRotatingFileHandler based on the configurations set in the file or the default configurations.
        Returns
        -------
        The TimedRotatingFileHandler instance.
        """
        if self.handler is not None:
            return self.handler
        else:
            if conf["full_path"]:
                path = conf["dir"]
                full_path = [conf["dir"], conf["name"]]
            else:
                path = get_full_path(conf["dir"])
                full_path = conf["dir"] + [conf["name"]]

            os.makedirs(path, exist_ok=True)
            full_path = get_full_path(full_path)
            handler = TimedRotatingFileHandler(
                full_path,
                when=conf["rotate_unit"],
                interval=conf["rotate_time"],
                backupCount=conf["backup_count"],
            )
            handler.setFormatter(
                logging.Formatter(
                    "[%(levelname)s] - %(asctime)s - %(name)s - %(funcName)s - %(lineno)d : %(message)s"
                )
            )
            handler.setLevel(logging.DEBUG)
            self.handler = handler
            return handler


def __configure(conf: dict[str, Any]):
    """
    Sets the dictionary of configurations based on the input configuration dictionary.
    Returns
    -------
    None
    """
    if len(conf) == 0:
        return configurations
    else:
        for key, value in conf.items():
            if key in configurations.keys():
                configurations[key] = value
        return configurations


def _get_console_handler(conf: dict[str, Any]) -> logging.StreamHandler:
    """
    Creates a Stream Handler for the Standard Output, with the same format as provided in configurations and at a
    Log Level of WARNING.
    Returns
    -------
    The StreamHandler for the console.
    """
    console_handler = ConsoleHandlerFactory().get_handler(conf)
    return console_handler


def _get_rotating_file_handler(conf: dict[str, Any]) -> TimedRotatingFileHandler:
    """
    Creates and returns a TimedRotatingFileHandler for a logger.
    Returns
    -------
    The TimedRotatingFileHandler.
    """
    timed_rotating_handler = TimedRotatingFileHandlerFactory().get_handler(conf)
    return timed_rotating_handler


def get_logger(conf: dict[str, Any] = {}) -> logging.Logger:
    """
    Function that creates and returns a logger if logging is enabled in the application.
    Returns
    -------
    A Logger object.
    """
    configurations = __configure(conf)
    if configurations["enable"]:
        logger = logging.getLogger(configurations["name"])
        logger.setLevel(configurations["existent_log_levels"][configurations["level"]])
        logger.addHandler(_get_console_handler(configurations))
        if configurations["to_file"]:
            logger.addHandler(_get_rotating_file_handler(configurations))
        logger.propagate = False
        return logger

    return logging.getLogger()


def get_full_path(path: list[str]) -> str:
    """
    Returns the full path, OS-independent, to the provided path, since the ROOT DIRECTORY of the package.
    Parameters
    ----------
    path A list of directories to traverse.
    Returns
    -------
    The string with the full path.
    """
    return os.path.join(ROOT_DIR, *path)
