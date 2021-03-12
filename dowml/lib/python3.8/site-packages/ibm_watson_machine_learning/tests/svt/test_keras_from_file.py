import unittest
import logging
from preparation_and_cleaning import *

class TestWMLClientWithKeras2CoreML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    converted_model_path = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithKeras2CoreML.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'core_ml', 'keras', 'mnistCNN.h5.tgz')

    def test_01_service_instance_details(self):
        TestWMLClientWithKeras2CoreML.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithKeras2CoreML.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):

        self.logger.info("Publishing Keras model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {
                        self.client.repository.ModelMetaNames.NAME: "Core ML - keras mnist model",
                        self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "tensorflow",
                        self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "1.5",
                        self.client.repository.ModelMetaNames.RUNTIME_NAME: "python",
                        self.client.repository.ModelMetaNames.RUNTIME_VERSION: "3.5",
                        self.client.repository.ModelMetaNames.FRAMEWORK_LIBRARIES: [{'name':'keras', 'version': '2.1.3'}],
                        self.client.repository.ModelMetaNames.EVALUATION_METHOD: "binary",
                        self.client.repository.ModelMetaNames.EVALUATION_METRICS: [{'name': 'accuracy', 'value': 0.8, 'threshold': 0.7}]
        }

        published_model_details = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithKeras2CoreML.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithKeras2CoreML.model_url = self.client.repository.get_model_url(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithKeras2CoreML.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithKeras2CoreML.model_url))
        self.assertIsNotNone(TestWMLClientWithKeras2CoreML.model_uid)
        self.assertIsNotNone(TestWMLClientWithKeras2CoreML.model_url)

    def test_04_publish_model_details(self):
        details_models = self.client.repository.get_model_details()
        TestWMLClientWithKeras2CoreML.logger.debug("All models details: " + str(details_models))
        self.assertTrue("Core ML - keras mnist model" in str(details_models))

    def test_05_model_libraries(self):
        import requests

        endpoint = self.wml_credentials['url']+'/v3/ml_assets/models/' + TestWMLClientWithKeras2CoreML.model_uid + '/versions'

        response_get = requests.get(endpoint, headers=self.client._get_headers())
        print(str(response_get.text))
        self.assertTrue('2.1.3' in str(response_get.text))

    def test_10_delete_model(self):
        self.client.repository.delete(TestWMLClientWithKeras2CoreML.model_uid)


if __name__ == '__main__':
    unittest.main()
