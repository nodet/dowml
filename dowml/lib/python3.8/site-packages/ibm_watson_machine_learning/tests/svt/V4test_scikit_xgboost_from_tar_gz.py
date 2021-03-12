import unittest
import os
from sklearn.datasets import load_svmlight_file
import logging
from preparation_and_cleaning import *


class TestWMLClientWithXGBoost(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithXGBoost.logger.info("Service Instance: setting up credentials")
        # reload(site)

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'scikit_xgboost_model.tar.gz')

    def test_01_service_instance_details(self):
        TestWMLClientWithXGBoost.logger.info("Check client ...")
        self.assertTrue(type(self.client).__name__ == 'APIClient')

        TestWMLClientWithXGBoost.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()

        TestWMLClientWithXGBoost.logger.debug(details)
        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_model(self):
        TestWMLClientWithXGBoost.logger.info("Publishing scikit-xgboost model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "SparkModel",
            self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.20",
            self.client.repository.ModelMetaNames.RUNTIME_UID: "scikit-learn_0.20-py3.6"
                       }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithXGBoost.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithXGBoost.logger.info("Published model ID:" + str(TestWMLClientWithXGBoost.model_uid))
        self.assertIsNotNone(TestWMLClientWithXGBoost.model_uid)

    def test_03_publish_model_details(self):
        TestWMLClientWithXGBoost.logger.info("Get published model details ...")
        details = self.client.repository.get_details(self.model_uid)

        TestWMLClientWithXGBoost.logger.debug("Model details: " + str(details))

    def test_04_create_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Create deployment ...")
        global deployment
        deployment = self.client.deployments.create(self.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithXGBoost.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithXGBoost.logger.info("Online deployment: " + str(deployment))
        TestWMLClientWithXGBoost.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestWMLClientWithXGBoost.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))


    # def test_06_update_model_version(self):
    #     deployment_details = self.client.deployments.get_details(TestWMLClientWithXGBoost.deployment_uid)
    #
    #     self.client.deployments.update(TestWMLClientWithXGBoost.deployment_uid)
    #     new_deployment_details = self.client.deployments.get_details(TestWMLClientWithXGBoost.deployment_uid)
    #
    #     self.assertNotEquals(deployment_details['entity']['deployed_version']['guid'], new_deployment_details['entity']['deployed_version']['guid'])

    def test_07_get_deployment_details(self):
        TestWMLClientWithXGBoost.logger.info("Get deployment details ...")
        deployment_details = self.client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

        TestWMLClientWithXGBoost.logger.debug("Online deployment: " + str(deployment_details))
        TestWMLClientWithXGBoost.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertIsNotNone(TestWMLClientWithXGBoost.deployment_uid)

    def test_08_score(self):
        TestWMLClientWithXGBoost.logger.info("Online model scoring ...")
        (X, _) = load_svmlight_file(os.path.join('artifacts', 'agaricus.txt.test'))
        scoring_data = {
            self.client.deployments.ScoringMetaNames.INPUT_DATA : [
                {
                    'values': [list(X.toarray()[0, :])]
                }
            ]
        }

        TestWMLClientWithXGBoost.logger.debug("Scoring data: {}".format(scoring_data))
        predictions = self.client.deployments.score(TestWMLClientWithXGBoost.deployment_uid, scoring_data)

        TestWMLClientWithXGBoost.logger.debug("Prediction: " + str(predictions))
        self.assertTrue(("prediction" in str(predictions)) and ("probability" in str(predictions)))

    def test_09_delete_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Delete model deployment ...")
        self.client.deployments.delete(TestWMLClientWithXGBoost.deployment_uid)

    def test_10_delete_model(self):
        TestWMLClientWithXGBoost.logger.info("Delete model ...")
        self.client.repository.delete(TestWMLClientWithXGBoost.model_uid)

if __name__ == '__main__':
    unittest.main()
