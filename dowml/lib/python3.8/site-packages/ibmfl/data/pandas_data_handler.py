"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
import abc
import logging

from ibmfl.data.data_handler import DataHandler

logger = logging.getLogger(__name__)


class PandasDataHandler(DataHandler):
    """
    Base class to load and pre-process data.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    @abc.abstractmethod
    def get_data(self):
        """
        Read data and return as Pandas data frame.

        :return: A dataset structure
        :rtype: `pandas.core.frame.DataFrame`
        """

    @abc.abstractmethod
    def get_dataset_info(self, **kwargs):
        """
        Read and extract data information

        :return: some information about the dataset (i.e. a dictionary that contains the list of features)
        :rtype: `dict`
        """
        raise NotImplemented

    def get_min(self, dp_flag=False, **kwargs):
        """
        Assuming the dataset is loaded as type `pandas.DataFrame`, and
        has shape(num_samples, num_features).

        :param dp_flag: Flag for differential private answer. By default is \
        set to False.
        :type dp_flag: `boolean`
        :param kwargs: Dictionary of differential privacy arguments \
        for computing the minimum value of each feature across all samples, \
        e.g., epsilon and delta, etc.
        :type kwargs: `dict`
        :return: A vector of shape (1, num_features) stores the minimum value \
        of each feature across all samples.
        :rtype: `pandas.Series` where each entry matches the original type \
        of the corresponding feature.
        """
        train_data, (_) = self.get_data()

        if not dp_flag:
            logger.info('Calculating minimum values.')
            min_vec = train_data.min(axis=0)
            # TODO dp minimum calculation
        return min_vec
