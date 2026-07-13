from configparser import ConfigParser
from core.src.common.constants import DEFAULT_CONFIG_PATH


class KYCConfigParser:
    """

    """
    def __init__(self):
        pass

    def get_config(self):
        """

        Returns:

        """
        config = ConfigParser()
        config.read(DEFAULT_CONFIG_PATH)
        return config