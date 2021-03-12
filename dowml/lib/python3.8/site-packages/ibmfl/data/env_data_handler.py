"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
"""
Module to where data handler are implemented.
"""
import abc

from ibmfl.data.data_handler import DataHandler
from ibmfl.data.env_spec import EnvHandler


class EnvDataHandler(DataHandler):
    """
    Base class to load data and  environment for reinforcement learning.
    """

    @abc.abstractmethod
    def get_data(self, **kwargs):
        """
        Read train data and test data for reinforcement learning
        :return:
        """

    @abc.abstractmethod
    def get_env_class_ref(self) -> EnvHandler:
        """
           Get environment reference for RL trainer, the instance is created in
           model class as part of trainer initialization
        """
