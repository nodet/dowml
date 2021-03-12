"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
import logging

from ibmfl.exceptions import LocalTrainingException, \
    ModelUpdateException

logger = logging.getLogger(__name__)


class LocalTrainingHandler():

    def __init__(self, fl_model, data_handler, hyperparams=None, **kwargs):
        """
        Initialize LocalTrainingHandler with fl_model, data_handler

        :param fl_model: model to be trained
        :type fl_model: `model.FLModel`
        :param data_handler: data handler that will be used to obtain data
        :type data_handler: `DataHandler`
        :param hyperparams: Hyperparameters used for training.
        :type hyperparams: `dict`
        :param kwargs: Additional arguments to initialize a local training \
        handler, e.g., a crypto library object to help with encryption and \
        decryption.
        :type kwargs: `dict`
        :return None
        """
        self.fl_model = fl_model
        self.data_handler = data_handler
        self.hyperparams = hyperparams

    def update_model(self, model_update):
        """
        Update local model with model updates received from FusionHandler

        :param model_update: ModelUpdate
        :type model_update: `ModelUpdate`
        """
        try:
            if model_update is not None:
                self.fl_model.update_model(model_update)
                logger.info('Local model updated.')
            else:
                logger.info('No model update was provided.')
        except Exception as ex:
            raise LocalTrainingException('No query information is provided',
                                         str(ex))

    def train(self,  fit_params=None):
        """
        Train locally using fl_model. At the end of training, a
        model_update with the new model information is generated and
        send through the connection.

        :param fit_params: (optional) Query instruction from aggregator
        :type fit_params: `dict`
        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        train_data, (_) = self.data_handler.get_data()

        self.update_model(fit_params.get('model_update'))

        logger.info('Local training started...')

        self.fl_model.fit_model(train_data, fit_params)

        update = self.fl_model.get_model_update()
        logger.info('Local training done, generating model update...')

        return update

    def save_model(self, payload=None):
        """
        Save the local model.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of save model request
        :rtype: `boolean`
        """
        status = False
        try:
            self.fl_model.save_model()
            status = True
        except Exception as ex:
            logger.error("Error occurred while saving local model")
            logger.exception(ex)

        return status

    def sync_model(self, payload=None):
        """
        Update the local model with global ModelUpdate received
        from the Aggregator.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Status of sync model request
        :rtype: `boolean`
        """
        status = False
        if payload is None or 'model_update' not in payload:
            raise ModelUpdateException(
                "Invalid Model update request aggregator")
        try:

            model_update = payload['model_update']
            self.fl_model.update_model(model_update=model_update)
        except Exception as ex:
            logger.error("Exception occurred while sync model")
            logger.exception(ex)

        return status

    def eval_model(self, payload=None):
        """
        Evaluate the local model based on the local test data.

        :param payload: data payload received from Aggregator
        :type payload: `dict`
        :return: Dictionary of evaluation results
        :rtype: `dict`
        """

        (_), test_dataset = self.data_handler.get_data()
        try:
            evaluations = self.fl_model.evaluate(test_dataset)

        except Exception as ex:
            logger.error("Expecting the test dataset to be of type tuple. "
                         "However, test dataset is of type "
                         + str(type(test_dataset)))
            logger.exception(ex)

        logger.info(evaluations)
        return evaluations
