import unittest
import os
import logging
from preparation_and_cleaning import *


class TestWMLClientWithPMML(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithPMML.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'iris_chaid.xml')

    def test_01_service_instance_details(self):
        TestWMLClientWithPMML.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        self.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithPMML.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):

        self.logger.info("Publishing PMML model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "LOCALLY created iris prediction model",
            self.client.repository.ModelMetaNames.TYPE: "pmml_4.2.1",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "pmml_4.2.1"
                       }
        published_model_details = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithPMML.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithPMML.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestWMLClientWithPMML.model_uid))
        self.logger.info("Published model URL:" + str(TestWMLClientWithPMML.model_url))
        self.assertIsNotNone(TestWMLClientWithPMML.model_uid)
        self.assertIsNotNone(TestWMLClientWithPMML.model_url)

    # def test_03_download_model(self):
    #     try:
    #         os.remove('download_test_url')
    #     except OSError:
    #         pass
    #
    #     self.client.repository.download(TestWMLClientWithPMML.model_uid, filename='download_test_url')
    #     self.assertRaises(WMLClientError, self.client.repository.download, TestWMLClientWithPMML.model_uid, filename='download_test_url')

    def test_04_publish_model_details(self):
        details = self.client.repository.get_details(self.model_uid)
        print(details)
        TestWMLClientWithPMML.logger.debug("Model details: " + str(details))
        self.assertTrue("LOCALLY created iris prediction model" in str(details))

        details_models = self.client.repository.get_model_details()
        TestWMLClientWithPMML.logger.debug("All models details: " + str(details_models))
        self.assertTrue("LOCALLY created iris prediction model" in str(details_models))

    def test_05_create_deployment(self):
        deployment = self.client.deployments.create(artifact_uid=TestWMLClientWithPMML.model_uid,meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})
        TestWMLClientWithPMML.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithPMML.logger.info("Online deployment: " + str(deployment))
        self.assertTrue(deployment is not None)
        TestWMLClientWithPMML.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestWMLClientWithPMML.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_06_get_deployment_details(self):
        deployment_details = self.client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_07_get_deployment_details_using_uid(self):
        deployment_details = self.client.deployments.get_details(TestWMLClientWithPMML.deployment_uid)
        print(deployment_details)
        self.assertIsNotNone(deployment_details)

    def test_08_score(self):
        scoring_data = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA:  [
                {
                    'fields': ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width'],
                    'values': [[5.1, 3.5, 1.4, 0.2]]
                }
            ]
        }
        predictions = self.client.deployments.score(TestWMLClientWithPMML.deployment_uid, scoring_data)
        print(predictions)
        predictions_fields = len(predictions)
        self.assertTrue(predictions_fields>0)

    def test_09_delete_deployment(self):
        self.client.deployments.delete(TestWMLClientWithPMML.deployment_uid)

    def test_10_delete_model(self):
        self.client.repository.delete(TestWMLClientWithPMML.model_uid)

if __name__ == '__main__':
    unittest.main()
