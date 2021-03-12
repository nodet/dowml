import unittest
import logging
from preparation_and_cleaning import *
from ibm_watson_machine_learning.utils import delete_directory, extract_mlmodel_from_archive
from models_preparation import *
from sys import platform


class TestWMLClientWithXGBoost2CoreML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    converted_model_path = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithXGBoost2CoreML.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'core_ml', 'xgboost', 'xgboost_model.tar.gz')

    def test_01_service_instance_details(self):
        TestWMLClientWithXGBoost2CoreML.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithXGBoost2CoreML.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):
        TestWMLClientWithXGBoost2CoreML.logger.info("Publishing xgboost model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.AUTHOR_NAME: "IBM",
                       self.client.repository.ModelMetaNames.NAME: "Agaricus prediction model",
                       self.client.repository.ModelMetaNames.FRAMEWORK_NAME: "xgboost",
                       self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "0.7"
                       }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithXGBoost2CoreML.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithXGBoost2CoreML.logger.info("Published model ID:" + str(TestWMLClientWithXGBoost2CoreML.model_uid))
        self.assertIsNotNone(TestWMLClientWithXGBoost2CoreML.model_uid)

    def test_04_publish_model_details(self):
        details_models = self.client.repository.get_model_details(TestWMLClientWithXGBoost2CoreML.model_uid)
        TestWMLClientWithXGBoost2CoreML.logger.debug("Model details: " + str(details_models))
        self.assertTrue("Agaricus prediction model" in str(details_models))

    def test_05_convert_2_coreml(self):
        self.logger.info("Converting xgboost model to core ml ...")
        deployment_details = self.client.deployments.create(TestWMLClientWithXGBoost2CoreML.model_uid, deployment_type='virtual')
        self.assertTrue('DEPLOY_SUCCESS' in str(deployment_details))
        TestWMLClientWithXGBoost2CoreML.deployment_uid = self.client.deployments.get_uid(deployment_details)

        print('Deployment UID: ' + TestWMLClientWithXGBoost2CoreML.deployment_uid)

    def test_06_load_CoreML_model(self):
        # Load the model
        import coremltools

        filepath = self.client.deployments.download(TestWMLClientWithXGBoost2CoreML.deployment_uid)
        print('Downloaded model path: ' + filepath)

        TestWMLClientWithXGBoost2CoreML.converted_model_path = extract_mlmodel_from_archive(filepath,
                                                                                          TestWMLClientWithXGBoost2CoreML.model_uid)

        loaded_model = coremltools.models.MLModel(TestWMLClientWithXGBoost2CoreML.converted_model_path)
        loaded_model.short_description = 'this is a test model'
        self.assertTrue('this is a test model' in str(loaded_model.short_description))

        if not platform in ['linux']:

            values = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0,
                      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                      0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]

            data = {}
            i = 0

            for x in values:
                data['f' + str(i)] = x
                i += 1

            scoring_data = data
            predictions = loaded_model.predict(scoring_data)
            print(str(predictions))
            self.assertTrue('5' in str(predictions))

    def test_10_delete_model(self):
        delete_directory(TestWMLClientWithXGBoost2CoreML.model_uid)
        self.client.repository.delete(TestWMLClientWithXGBoost2CoreML.model_uid)


if __name__ == '__main__':
    unittest.main()
