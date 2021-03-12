import unittest
import os
import logging
from preparation_and_cleaning import *


class TestWMLClientWithKeras(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    x_test = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithKeras.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'keras_model.tgz')

    def test_00_check_client_version(self):
        TestWMLClientWithKeras.logger.info("Check client version...")

        self.logger.info("Getting version ...")
        version = self.client.version
        TestWMLClientWithKeras.logger.debug(version)
        self.assertTrue(len(version) > 0)

    def test_01_service_instance_details(self):
        TestWMLClientWithKeras.logger.info("Check client ...")
        self.assertTrue(type(self.client).__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithKeras.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):
        TestWMLClientWithKeras.logger.info("Creating keras model ...")
        from keras.datasets import mnist

        from keras import backend as K

        batch_size = 128
        num_classes = 10
        epochs = 1

        # input shape
        img_rows, img_cols = 28, 28

        # samples to train
        num_train_samples = 500

#        print(K._backend)

        # prepare train and test datasets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        if K.image_data_format() == 'channels_first':
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        else:
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_test = x_test.astype('float32')
        x_test /= 255
        print(x_test.shape[0], 'test samples')

        TestWMLClientWithKeras.x_test = x_test

        self.logger.info("Publishing keras model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "KerasModel-gzip",
            self.client.repository.ModelMetaNames.TYPE: "keras_2.2.5",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "tensorflow_1.15-py3.6"
                       }
        published_model_details = self.client.repository.store_model(model=TestWMLClientWithKeras.model_path,
                                                                     meta_props=model_props)  # , training_data=digits.data, training_target=digits.target)
        TestWMLClientWithKeras.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithKeras.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithKeras.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithKeras.model_url))
        self.assertIsNotNone(TestWMLClientWithKeras.model_uid)
        self.assertIsNotNone(TestWMLClientWithKeras.model_url)

    # def test_03_download_model(self):
    #     TestWMLClientWithKeras.logger.info("Download model")
    #     try:
    #         os.remove('download_test_url')
    #     except OSError:
    #         pass
    #
    #     try:
    #         file = open('download_test_uid', 'r')
    #     except IOError:
    #         file = open('download_test_uid', 'w')
    #         file.close()
    #
    #     self.client.repository.download(TestWMLClientWithKeras.model_uid, filename='download_test_url')
    #     self.assertRaises(WMLClientError, self.client.repository.download, TestWMLClientWithKeras.model_uid,
    #                       filename='download_test_uid')

    def test_04_get_details(self):
        TestWMLClientWithKeras.logger.info("Get model details")
        details = self.client.repository.get_details(self.model_uid)
        TestWMLClientWithKeras.logger.debug("Model details: " + str(details))
        self.assertTrue("KerasModel-gzip" in str(details))

    def test_05_create_deployment(self):
        TestWMLClientWithKeras.logger.info("Create deployments")
        deployment = self.client.deployments.create(TestWMLClientWithKeras.model_uid, meta_props={
            self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})
        TestWMLClientWithKeras.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithKeras.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithKeras.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestWMLClientWithKeras.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_06_get_deployment_details(self):
        TestWMLClientWithKeras.logger.info("Get deployment details")
        deployment_details = self.client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_07_get_deployment_details_using_uid(self):
        TestWMLClientWithKeras.logger.info("Get deployment details using uid")
        deployment_details = self.client.deployments.get_details(TestWMLClientWithKeras.deployment_uid)
        self.assertIsNotNone(deployment_details)

    def test_08_score(self):
        TestWMLClientWithKeras.logger.info("Score model")
        x_score_1 = TestWMLClientWithKeras.x_test[23].tolist()
        x_score_2 = TestWMLClientWithKeras.x_test[32].tolist()
        scoring_payload = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA : [
                {
                    'values': [x_score_1, x_score_2]
                }
            ]
        }
        predictions = self.client.deployments.score(TestWMLClientWithKeras.deployment_uid, scoring_payload)
        self.assertTrue("prediction" in str(predictions))

    def test_09_delete_deployment(self):
        TestWMLClientWithKeras.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithKeras.deployment_uid)

    def test_10_delete_model(self):
        TestWMLClientWithKeras.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithKeras.model_uid)


if __name__ == '__main__':
    unittest.main()
