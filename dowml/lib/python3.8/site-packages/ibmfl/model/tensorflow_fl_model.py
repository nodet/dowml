"""
IBM Confidential
OCO Source Materials
5737-H76, 5725-W78, 5900-A1R
(c) Copyright IBM Corp. 2020 All Rights Reserved.
The source code for this program is not published or otherwise divested of its trade secrets,
irrespective of what has been deposited with the U.S. Copyright Office.
"""
import logging
import time
import numpy as np
import tensorflow as tf

# if tf.__version__ != "2.1.0":
# raise ImportError("This function requires TensorFlow v2.1.0.")

from ibmfl.util import config
from ibmfl.model.fl_model import FLModel
from ibmfl.model.model_update import ModelUpdate
from ibmfl.exceptions import FLException, LocalTrainingException, ModelException

logger = logging.getLogger(__name__)


class TensorFlowFLModel(FLModel):
    """
    Wrapper class for importing tensorflow models.
    """

    def __init__(self, model_name, model_spec, tf_model=None, **kwargs):
        """
        Create a `TensorFlowFLModel` instance from a tensorflow model.\
        If `tf_model` is provided, it will use it; otherwise it will take\
        the model_spec to create the model.\
        Assumes the `tf_model` passed as argument is compiled.

        :param model_name: String specifying the type of model e.g., tf_CNN
        :type model_name: `str`
        :param model_spec: Specification of the `tf_model`
        :type model_spec: `dict`
        :param tf_model: Compiled TensorFlow model.
        :type tf_model: `tf.keras.Model`
        """

        super().__init__(model_name, model_spec, **kwargs)
        if tf_model is None:
            if model_spec is None or (not isinstance(model_spec, dict)):
                raise ValueError('Initializing model requires '
                                 'a model specification or '
                                 'compiled TensorFlow model. '
                                 'None was provided')
            # In this case we need to recreate the model from model_spec
            self.model = self.load_model_from_spec(model_spec)
        else:
            if not issubclass(type(tf_model), tf.keras.Model):
                raise ValueError('Compiled TensorFlow model needs to be '
                                 'provided of type `tensorflow.keras.models`.'
                                 ' Type provided' + str(type(tf_model)))

            if self.use_gpu_for_training and self.num_gpus >= 1:
                strategy = tf.distribute.MirroredStrategy()
                with strategy.scope():
                    self.model = tf_model
            else:
                self.model = tf_model

        # Default values for local training
        self.batch_size = 128
        self.epochs = 1
        self.steps_per_epoch = 100

    def fit_model(self, train_data, fit_params=None, **kwargs):
        """
        Fits current model with provided training data.

        :param train_data: Training data, a tuple\
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :param fit_params: (optional) Dictionary with hyperparameters\
        that will be used to call fit function.\
        Hyperparameter parameters should match  expected values\
        e.g., `epochs`, which specifies the number of epochs to be run.\
        If no `epochs` or `batch_size` are provided, a default value\
        will be used (1 and 128, respectively).
        :type fit_params: `dict`
        :return: None
        """
        # Initialized with default values
        batch_size = self.batch_size
        epochs = self.epochs
        steps_per_epoch = self.steps_per_epoch

        # extract hyperparams from fit_param
        if fit_params and ('hyperparams' in fit_params):
            hyperparams = fit_params['hyperparams']
            if hyperparams is not None and hyperparams.get('local') is not None and hyperparams.get('local').get('training') is not None :
                training_hp = hyperparams['local']['training']

                if 'batch_size' in training_hp:
                    batch_size = training_hp['batch_size']
                else:
                    # In this case, use default values.
                    logger.info('Using default hyperparameters: '
                                ' batch_size:' + str(self.batch_size))
                if 'epochs' in training_hp:
                    epochs = training_hp['epochs']
                else:
                    # In this case, use default values.
                    logger.info('Using default hyperparameters: '
                                ' epochs:' + str(self.epochs))

                if 'steps_per_epoch' in training_hp:
                    steps_per_epoch = training_hp.get('steps_per_epoch')

            else :
                # In this case, use default values.
                logger.info('Using default hyperparameters: '
                            'epochs:' + str(self.epochs) +
                            ' batch_size:' + str(self.batch_size))
        try:
            if type(train_data) is tuple and type(train_data[0]) is np.ndarray:
                # Extract x_train and y_train, by default,
                # label is stored in the last column
                x = train_data[0]
                y = train_data[1]
                self.model.fit(x, y, batch_size=batch_size, epochs=epochs)
            else:
                self.model.fit(train_data, epochs=epochs,
                               steps_per_epoch=steps_per_epoch)

        except Exception as e:
            logger.exception(str(e))
            if epochs is None:
                logger.exception('epochs need to be provided')

            raise LocalTrainingException(
                'Error occurred while performing model.fit')

    def update_model(self, model_update):
        """
        Update TensorFlow model with provided model_update, where model_update \
        should be generated according to \
        `TensorFlowFLModel.get_model_update()`. 

        :param model_update: `ModelUpdate` object that contains the weights \
        that will be used to update the model. 
        :type model_update: `ModelUpdate`
        :return: None
        """
        if isinstance(model_update, ModelUpdate):
            w = model_update.get("weights")
            self.model.set_weights(w)
        else:
            raise LocalTrainingException('Provided model_update should be of '
                                         'type ModelUpdate. '
                                         'Instead they are:' +
                                         str(type(model_update)))

    def get_model_update(self):
        """
        Generates a `ModelUpdate` object that will be sent to other entities.

        :return: ModelUpdate
        :rtype: `ModelUpdate`
        """
        w = self.model.get_weights()
        return ModelUpdate(weights=w)

    def predict(self, x, **kwargs):
        """
        Perform prediction for a batch of inputs. Note that for classification \
        problems, it returns the resulting probabilities.

        :param x: Samples with shape as expected by the model.
        :type x: `np.ndarray`
        :param kwargs: Dictionary of tf-specific arguments.
        :type kwargs: `dict`

        :return: Array of predictions
        :rtype: `np.ndarray`
        """
        return self.model.predict(x, **kwargs)

    def evaluate(self, test_dataset, batch_size=128, **kwargs):
        """
        Evaluates the model given testing data.

        :param test_dataset: Testing data, a tuple given in the form \
        (x_test, y_test) or a datagenerator of type `keras.utils.Sequence`,
        `keras.preprocessing.image.ImageDataGenerator`
        :type test_dataset: `np.ndarray` or `keras.utils.Sequence`, \
        `keras.preprocessing.image.ImageDataGenerator`
        :param batch_size: batch_size: Size of batches.
        :type batch_size: `int`
        :param kwargs: Dictionary of metrics available for the model
        :type kwargs: `dict`
        """
        if type(test_dataset) is tuple:
            x_test = test_dataset[0]
            y_test = test_dataset[1]

            metrics = self.model.evaluate(x_test, y_test,
                                          batch_size=batch_size,
                                          **kwargs)
        else:
            metrics = self.model.evaluate(test_dataset, **kwargs)

        names = self.model.metrics_names
        dict_metrics = {}

        if type(metrics) == list:
            for metric, name in zip(metrics, names):
                metric = metric.item()
                if name == 'accuracy':
                    dict_metrics['acc'] = round(metric, 2)
                
                dict_metrics[name] = metric
        else:
            dict_metrics[names[0]] = metrics
        return dict_metrics

    @staticmethod
    def load_model(file_name, custom_objects={}):
        """
        Loads a model from disk given the specified file_name

        :param file_name: Name of the file that contains the model to be loaded.
        :type file_name: `str`
        :return: TensorFlow model loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        try:
            model = tf.keras.models.load_model(file_name)
        except Exception as ex:
            logger.exception(str(ex))
            logger.error(
                'Loading model via tf.keras.models.load_model failed!')
        return model

    def save_model(self, filename=None):
        """
        Save a model to file in the format specific to the backend framework.

        :param filename: Name of the file where to store the model.
        :type filename: `str`
        :return: filename
        :rtype `string`
        """
        if filename is None:
            filename = '.'

        full_path = super().get_model_absolute_path(filename)
        self.model.save(full_path)
        logger.info('Model saved in path: %s.', full_path)
        return filename

    @staticmethod
    def model_from_json_via_tf_keras(json_file_name):
        """
        Loads a model architecture from disk via tf.keras \
        given the specified json file name.

        :param json_file_name: Name of the file that contains \
        the model architecture to be loaded.
        :type json_file_name: `str`
        :return: tf.keras model with only model architecture loaded to memory
        :rtype: `tf.keras.models.Model`
        """
        model = None
        json_file = open(json_file_name, 'r')
        f = json_file.read()
        json_file.close()
        try:
            model = tf.keras.models.model_from_json(f)
        except Exception as ex:
            logger.error(
                'Loading model via tf.keras.models.model_from_json failed! ')

        return model

    def load_model_from_spec(self, model_spec):
        """
        Loads model from provided model_spec, where model_spec is a `dict` \
        that contains the following items: \
            'model_definition': the path where the tf model is stored, \
                usually in a `SavedModel` format. 
        :return: model
        :rtype: `keras.models.Model`
        """
        if 'model_definition' in model_spec:
            try:
                model_file = model_spec['model_definition']
                model_absolute_path = config.get_absolute_path(model_file)

                custom_objects = {}
                if self.use_gpu_for_training:
                    strategy = tf.distribute.MirroredStrategy()
                    with strategy.scope():
                        model = TensorFlowFLModel.load_model(
                            model_absolute_path, custom_objects=custom_objects)
                else:
                    model = TensorFlowFLModel.load_model(
                        model_absolute_path, custom_objects=custom_objects)

            except Exception as ex:
                logger.exception(str(ex))
                raise FLException('Failed to load TensorFlow model!')
        else:
            # Load architecture from json file
            try:
                model = TensorFlowFLModel.model_from_json_via_tf_keras(
                    model_spec['model_architecture'])

                if model is None:
                    logger.error(
                        'An acceptable compiled model should be of type '
                        'tensorflow.keras.models!')
            except Exception as ex:
                logger.error(str(ex))
                raise FLException(
                    'Unable to load the provided uncompiled model!')

            # Load weights from provided path
            if 'model_weights' in model_spec:
                model.load_weights(model_spec['model_weights'])

            if 'compile_model_options' in model_spec:
                # Load compile options:
                compiled_options = model_spec['compile_model_options']
                optimizer = compiled_options.get('optimizer')
                loss = compiled_options.get('loss')
                metrics = compiled_options.get('metrics')
                metrics = [metrics] if not isinstance(
                    metrics, list) else metrics
                try:
                    if self.use_gpu_for_training:
                        strategy = tf.distribute.MirroredStrategy()
                        with strategy.scope():
                            model.compile(optimizer=optimizer,
                                          loss=loss,
                                          metrics=metrics)
                    else:
                        model.compile(optimizer=optimizer,
                                      loss=loss,
                                      metrics=metrics)
                except Exception as ex:
                    logger.exception(str(ex))
                    logger.exception(
                        'Failed to compiled the TensorFlow.keras model.')
            else:
                raise ModelException(
                    'Failed to compile keras model, no compile options provided.')
        return model

    def get_gradient(self, train_data):
        """
        Compute the gradient with the provided dataset at the current local \
        model's weights.

        :param train_data: Training data, a tuple \
        given in the form (x_train, y_train).
        :type train_data: `np.ndarray`
        :return: gradients
        :rtype: `list` of `tf.Tensor`
        """
        try:
            x, y = train_data[0], train_data[1]
        except Exception as ex:
            logger.exception(str(ex))
            raise FLException('Provided dataset has incorrect format. '
                              'It should be a tuple in the form of '
                              '(x_train, y_train).')
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.model.loss(y, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        return gradients

    def is_fitted(self):
        """
        Return a boolean value indicating if the model is fitted or not. \
        In particular, check if the tensorflow model has weights. \
        If it has, return True; otherwise return false. 

        :return: res
        :rtype: `bool`
        """
        try:
            self.model.get_weights()
        except Exception:
            return False
        return True
