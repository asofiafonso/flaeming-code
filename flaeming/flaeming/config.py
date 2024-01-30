import logging
import os

import yaml


logger = logging.getLogger(__name__)


class Singleton(type):
    """
    Singleton Class to ensure unique configuration
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Configuration(metaclass=Singleton):
    """
    Class to 'set' and 'get' configurations, with 'Singleton' as its metaclass.
    """

    default_config_path = f"{os.path.dirname(__file__)}/config.yaml"
    initiated = False

    def __init__(self, config_path: str = None):
        """Initializes the configuration object.

        :param config_path: The path to the configuration file. If None, loads
            local configuration, defaults to None
        :type config_path: str, optional

        :raises IOError: If path is provided but file is not found on system.
        """
        if config_path is None:
            self.path = self.default_config_path
        elif os.path.isfile(config_path):
            self.path = config_path
        else:
            logger.error(f"File not found: {config_path}")
            raise IOError
        self.data = self.load_config(self.path)
        Configuration.initiated = True

    def __merge_dictionaries(self, default_dict: dict, updated_dict: dict) -> dict:
        """Utility method two merge two nested dictionaries.

        :param default_dict: Default dictionary to be updated
        :type default_dict: dict
        :param updated_dict: Subset of dictionary with keys to be added/updated.
        :type updated_dict: dict
        :return: A combination of both dictionaries. If the same keys are in both,
            the value from the updated dictionary is kept.
        :rtype: dict
        """
        for key, value in updated_dict.items():
            if (
                isinstance(value, dict)
                and key in default_dict.keys()
                and isinstance(default_dict[key], dict)
            ):
                self.__merge_dictionaries(default_dict[key], updated_dict[key])
            else:
                default_dict[key] = updated_dict[key]
        return default_dict

    def load_yaml(self, path: str) -> dict:
        """Small wrapper to load YAML configuration files.

        :param path: The path to the .yaml file.
        :type path: str
        :return: The configuration information from the .yaml file as a dictionary.
        :rtype: dict
        """
        logger.debug(f"Loading config {path}")
        with open(path, "r", encoding="utf8") as stream:
            data = yaml.safe_load(stream)
            return data

    def load_config(self, config_path: str) -> dict:
        """Loads YAML configuration file.

        :param config_path: The path to the configuration file
        :type config_path: str

        :return: A dictionary with the loaded configurations
        :rtype: dict
        """
        if config_path == self.default_config_path:
            return self.load_yaml(config_path)
        else:
            default_config = self.load_yaml(self.default_config_path)
            custom_config = self.load_yaml(config_path)
            return self.__merge_dictionaries(default_config, custom_config)

    def load_nested_dictionary(self, keys: list[str]) -> str:
        """Given a set of keys separated by /, returns
        the nested dictionary equivalent.

        :param keys: The keys to search for in a list
        :type keys: list[str]

        :return: The value of the dictionary with the nested keys.
        :rtype: str
        """
        key_construct = "".join([f"['{k}']" for k in keys])
        return eval(f"self.data{key_construct}")

    def get(self, keys: str) -> str:
        """
        To get configuration values for a given configuration key.
        If the key is not found in the _data, default configuration value is returned.

        :paramg key: The key to get from the configuration.
        :type key: str
        :return: Value of the key.
        :rtype: str
        """
        return self.load_nested_dictionary(keys.split("/"))

    def set(self, keys: str, value: str):
        """
        To set configuration values for a given configuration key.

        :paramg key: Configuration key.
        :type key: str
        :return: Configuration value.
        :rtype: str
        """
        search_keys = keys.split("/")
        parent_dict = self.load_nested_dictionary(search_keys[:-1])
        parent_dict[search_keys[-1]] = value
        return

    def reset(self):
        """
        To reset configuration data with the default dictionary.
        """
        self.data = self.load_config(self.path)
