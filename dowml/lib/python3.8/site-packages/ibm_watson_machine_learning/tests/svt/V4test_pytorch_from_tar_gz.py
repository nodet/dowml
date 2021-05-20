import unittest
import os
from sklearn.datasets import load_svmlight_file
import logging
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient
import numpy as np


class TestWMLClientWithPyTorch(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    space_name = 'tests_sdk_space'
    space_id = None
    model_path = os.path.join('.', 'svt', 'artifacts', 'mnist_pytorch.tar.gz')
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls) -> None:
        """
        Load WML credentials from config.ini file based on ENV variable.
        """

        cls.wml_credentials = get_wml_credentials()

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)

        if not cls.wml_client.ICP:
            cls.cos_credentials = get_cos_credentials()
            cls.cos_endpoint = cls.cos_credentials.get('endpoint_url')
            cls.cos_resource_instance_id = cls.cos_credentials.get('resource_instance_id')

        cls.wml_client = APIClient(wml_credentials=cls.wml_credentials)
        cls.project_id = cls.wml_credentials.get('project_id')

    def test_00a_space_cleanup(self):
        space_cleanup(self.wml_client,
                      get_space_id(self.wml_client, self.space_name,
                                   cos_resource_instance_id=self.cos_resource_instance_id),
                      days_old=7)
        TestWMLClientWithPyTorch.space_id = get_space_id(self.wml_client, self.space_name,
                                                         cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_02_publish_model(self):
        TestWMLClientWithPyTorch.logger.info("Publishing pytorch model ...")

        self.wml_client.repository.ModelMetaNames.show()

        self.wml_client.software_specifications.list()
        sw_spec_id = self.wml_client.software_specifications.get_id_by_name("default_py3.7")

        model_props = {self.wml_client.repository.ModelMetaNames.NAME: "External pytorch model",
            self.wml_client.repository.ModelMetaNames.TYPE: "pytorch-onnx_1.3",
            self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                       }
        published_model = self.wml_client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithPyTorch.model_uid = self.wml_client.repository.get_model_uid(published_model)
        TestWMLClientWithPyTorch.logger.info("Published model ID:" + str(TestWMLClientWithPyTorch.model_uid))
        self.assertIsNotNone(TestWMLClientWithPyTorch.model_uid)

    def test_03_publish_model_details(self):
        TestWMLClientWithPyTorch.logger.info("Get published model details ...")
        details = self.wml_client.repository.get_details(self.model_uid)

        TestWMLClientWithPyTorch.logger.debug("Model details: " + str(details))

    def test_04_create_deployment(self):
        TestWMLClientWithPyTorch.logger.info("Create deployment ...")
        global deployment
        deployment = self.wml_client.deployments.create(self.model_uid, meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.wml_client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithPyTorch.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithPyTorch.logger.info("Online deployment: " + str(deployment))
        TestWMLClientWithPyTorch.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        TestWMLClientWithPyTorch.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))


    # def test_06_update_model_version(self):
    #     deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithXGBoost.deployment_uid)
    #
    #     self.wml_client.deployments.update(TestWMLClientWithXGBoost.deployment_uid)
    #     new_deployment_details = self.wml_client.deployments.get_details(TestWMLClientWithXGBoost.deployment_uid)
    #
    #     self.assertNotEquals(deployment_details['entity']['deployed_version']['guid'], new_deployment_details['entity']['deployed_version']['guid'])

    def test_07_get_deployment_details(self):
        TestWMLClientWithPyTorch.logger.info("Get deployment details ...")
        deployment_details = self.wml_client.deployments.get_details()
        self.assertTrue("Test deployment" in str(deployment_details))

        TestWMLClientWithPyTorch.logger.debug("Online deployment: " + str(deployment_details))
        TestWMLClientWithPyTorch.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertIsNotNone(TestWMLClientWithPyTorch.deployment_uid)

    def test_08_score(self):
        TestWMLClientWithPyTorch.logger.info("Online model scoring ...")
        # (X, _) = load_svmlight_file(os.path.join('svt', 'artifacts', 'mnist.npz'))
        dataset = np.load(os.path.join('svt', 'artifacts', 'mnist.npz'))
        X = dataset['x_test']

        score_0 = [X[0].tolist()]
        score_1 = [X[1].tolist()]

        scoring_data = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA : [
                {
                    'values': [score_0, score_1]
                }
            ]
        }

        TestWMLClientWithPyTorch.logger.debug("Scoring data: {}".format(scoring_data))
        predictions = self.wml_client.deployments.score(TestWMLClientWithPyTorch.deployment_uid, scoring_data)
        print(predictions)

        TestWMLClientWithPyTorch.logger.debug("Prediction: " + str(predictions))
        self.assertTrue(("prediction" in str(predictions)) and ("values" in str(predictions)))

    def test_09_delete_deployment(self):
        TestWMLClientWithPyTorch.logger.info("Delete model deployment ...")
        self.wml_client.deployments.delete(TestWMLClientWithPyTorch.deployment_uid)

    def test_10_delete_model(self):
        TestWMLClientWithPyTorch.logger.info("Delete model ...")
        self.wml_client.repository.delete(TestWMLClientWithPyTorch.model_uid)

if __name__ == '__main__':
    unittest.main()
