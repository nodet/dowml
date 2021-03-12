"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
import os
import logging
import logging.config

from ibmfl._version import __version__
from ibmfl.exceptions import InvalidConfigurationException

logger = logging.getLogger(__name__)


class FFLVersionFilter(logging.Filter):

    def filter(self, record):
        record.version = __version__
        return True


default_log_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "fl_std": {
            "format": "%(asctime)s | %(version)s | %(levelname)s | %(name)-50s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "filters": ['version_filter'],
            "level": "DEBUG",
            "formatter": "fl_std",
            "stream": "ext://sys.stdout"
        }

    },
    "loggers": {
        "ibmfl": {
            "level": "INFO",
            "handlers": [
                "console"
            ],
            "propagate": False
        }
    },
    "root": {
        "level": "INFO",
        "handlers": [
            "console"
        ]
    }
}


def configure_file_logging(filename, log_level=logging.INFO):

    dict_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "fl_std": {
                "format": "%(asctime)s | %(version)s | %(levelname)s | %(name)-50s | %(message)s"
            }
        },
        "handlers": {
            "WriteFile": {
                "class": "logging.FileHandler",
                "filters": ['version_filter'],
                "level": "DEBUG",
                "formatter": "fl_std",
                "filename": filename
            }

        },
        "loggers": {
            "ibmfl": {
                "level": "INFO",
                "handlers": [
                    "WriteFile"
                ],
                "propagate": False
            }
        }
    }

    result = configure_logging(dict_config, log_level=logging.INFO)
    return result


def configure_logging(config, log_level=logging.INFO):
    """
    configures logging for application based on the configuration file

    :param config: yaml file containing the definitions of formatter and handler
    :type config: `dict`
    :param log_level: should be a value from [DEBUG, INFO, WARNING, ERROR, CRITICAL]
            based on the required granularity
    :type log_level: `int`
    :return: a boolean object. False for default basic config True otherwise
    :rtype: `boolean`
    """

    if not config:
        config = default_log_config

    log_filters = {
        'version_filter': {
            '()': FFLVersionFilter
        }
    }
    if 'filters' in config:
        config['filters']['version_filter'] = log_filters['version_filter']
    else:
        config['filters'] = log_filters

    add_version_filter(config)
    result = False
    if config:
        try:
            logging.config.dictConfig(config)
            result = True
        except InvalidConfigurationException as ice:
            logging.error('Failed to load log configuration %s', ice)
            configure_basic_logging(log_level=log_level)
    else:
        configure_basic_logging(log_level=log_level)

    return result


def add_version_filter(config):
    """
    Add versioning filter to log config if not provided

    :param config: yaml file containing the definitions of formatter and handler
    :type config: `dict`
    :return: None
    """
    filters = ['version_filter']
    if 'handlers' in config:

        handlers = config['handlers']

        for key in handlers:
            handler = handlers.get(key)

            if 'filters' in handler:
                filt = handler.get('filters')
                if 'version_filter' not in filt:
                    filt.append('version_filter')
            else:
                handler['filters'] = filters


def configure_basic_logging(log_level=logging.INFO):
    """
    configures logging for the session based on pre defined format.
    default logging will be done on console.

    :param level: should be a value from [DEBUG, INFO, WARNING, ERROR, CRITICAL]
            based on the required granularity
    :type log_level: `int`
    :return: None
    """

    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # logging.config.dictConfig(default_log_config)
