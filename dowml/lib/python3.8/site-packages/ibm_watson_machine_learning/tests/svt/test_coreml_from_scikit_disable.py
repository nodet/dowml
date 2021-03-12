import unittest
import logging
from preparation_and_cleaning import *
from ibm_watson_machine_learning.utils import delete_directory, extract_mlmodel_from_archive
from models_preparation import *
from sys import platform

class TestWMLClientWithScikit2CoreML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    converted_model_path = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithScikit2CoreML.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        #self.model_path = os.path.join(os.getcwd(), 'artifacts', 'core_ml', 'sklearn', '')

    def test_01_service_instance_details(self):
        TestWMLClientWithScikit2CoreML.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithScikit2CoreML.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):
        TestWMLClientWithScikit2CoreML.logger.info("Creating scikit-learn model ...")

        model_data = create_scikit_learn_model_data('digits')
        predicted = model_data['prediction']

        TestWMLClientWithScikit2CoreML.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {
            self.client.repository.ModelMetaNames.NAME: "Hand written digits prediction model",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.19",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "scikit-learn_0.19-py3"
                      }

        published_model_details = self.client.repository.store_model(model=model_data['model'], meta_props=model_props, training_data=model_data['training_data'], training_target=model_data['training_target'])
        TestWMLClientWithScikit2CoreML.model_uid = self.client.repository.get_model_uid(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithScikit2CoreML.model_uid))
        self.assertIsNotNone(TestWMLClientWithScikit2CoreML.model_uid)

    def test_04_publish_model_details(self):
        details_models = self.client.repository.get_model_details(TestWMLClientWithScikit2CoreML.model_uid)
        print("All models details: " + str(details_models))
        self.assertTrue("Hand written digits prediction model" in str(details_models))

    def test_05_convert_2_coreml(self):
        self.logger.info("Converting scikit-learn model to core ml ...")

        deployment_details = self.client.deployments.create(TestWMLClientWithScikit2CoreML.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.VIRTUAL:{}})
        self.assertTrue('DEPLOY_SUCCESS' in str(deployment_details))
        TestWMLClientWithScikit2CoreML.deployment_uid = self.client.deployments.get_uid(deployment_details)
        print(TestWMLClientWithScikit2CoreML.deployment_uid)

    def test_06_load_CoreML_model(self):
        # Load the model
        import coremltools

        filepath = self.client.deployments.download(TestWMLClientWithScikit2CoreML.deployment_uid)
        print('Downloaded model path: ' + filepath)

        TestWMLClientWithScikit2CoreML.converted_model_path = extract_mlmodel_from_archive(filepath,
                                                                                          TestWMLClientWithScikit2CoreML.model_uid)

        loaded_model = coremltools.models.MLModel(TestWMLClientWithScikit2CoreML.converted_model_path)
        loaded_model.short_description = 'this is a test model'
        self.assertTrue('this is a test model' in str(loaded_model.short_description))

        if not platform in ['linux']:
            scoring_data = {'input': [0.0, 0.0, 5.0, 16.0, 16.0, 3.0, 0.0, 0.0, 0.0, 0.0, 9.0, 16.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 12.0, 15.0,
                                      2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 15.0, 16.0, 15.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 13.0, 16.0, 9.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 14.0, 12.0, 0.0, 0.0, 0.0, 0.0, 5.0, 12.0, 16.0, 8.0, 0.0, 0.0, 0.0, 0.0, 3.0, 15.0,
                                      15.0, 1.0, 0.0, 0.0]}

            predictions = loaded_model.predict(scoring_data)
            self.assertTrue('5' in str(predictions))

    def test_10_delete_model(self):
        delete_directory(TestWMLClientWithScikit2CoreML.model_uid)
        self.client.repository.delete(TestWMLClientWithScikit2CoreML.model_uid)


if __name__ == '__main__':
    unittest.main()
