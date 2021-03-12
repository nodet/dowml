import unittest
import logging
from preparation_and_cleaning import *


class TestWMLClientWithTensorflow(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithTensorflow.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'model_dir')

    def test_01_service_instance_details(self):
        TestWMLClientWithTensorflow.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithTensorflow.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithTensorflow.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_publish_local_model_in_repository(self):
        TestWMLClientWithTensorflow.logger.info("Saving trained model in repo ...")
        TestWMLClientWithTensorflow.logger.debug(self.model_path)

        self.client.repository.ModelMetaNames.show()

        model_meta_props = {self.client.repository.ModelMetaNames.NAME: "LOCALLY created agaricus prediction model",
                       self.client.repository.ModelMetaNames.TYPE: "tensorflow_1.15",
                       self.client.repository.ModelMetaNames.RUNTIME_UID: "tensorflow_1.15-py3.6"
                            }
        published_model_details = self.client.repository.store_model(model=self.model_path, meta_props=model_meta_props)
        TestWMLClientWithTensorflow.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestWMLClientWithTensorflow.logger.info("Published model ID:" + str(TestWMLClientWithTensorflow.model_uid))
        self.assertIsNotNone(TestWMLClientWithTensorflow.model_uid)

    def test_04_load_model(self):
        TestWMLClientWithTensorflow.logger.info("Load model from repository: {}".format(TestWMLClientWithTensorflow.model_uid))
        self.tf_model = self.client.repository.load(TestWMLClientWithTensorflow.model_uid)
        TestWMLClientWithTensorflow.logger.debug("TF type: {}".format(type(self.tf_model)))
        self.assertTrue(self.tf_model)

    def test_05_create_deployment(self):
        TestWMLClientWithTensorflow.logger.info("Create deployment")
        deployment_details = self.client.deployments.create(artifact_uid=TestWMLClientWithTensorflow.model_uid,
                                                            meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",
                                                                        self.client.deployments.ConfigurationMetaNames.ONLINE: {}})


        TestWMLClientWithTensorflow.deployment_uid = self.client.deployments.get_uid(deployment_details)
        TestWMLClientWithTensorflow.scoring_url = self.client.deployments.get_scoring_href(deployment_details)
        self.assertTrue('online' in str(deployment_details))

    def test_06_scoring(self):
        TestWMLClientWithTensorflow.logger.info("Score model")
        scoring_data = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 18, 18, 18,
                                 126, 136, 175, 26, 166, 255, 247, 127, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 30, 36, 94, 154, 170, 253,
                                 253, 253, 253, 253, 225, 172, 253, 242, 195, 64, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 49, 238, 253, 253, 253,
                                 253, 253, 253, 253, 253, 251, 93, 82, 82, 56, 39, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 219, 253,
                                 253, 253, 253, 253, 198, 182, 247, 241, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 80, 156, 107, 253, 253, 205, 11, 0, 43, 154, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 14, 1, 154, 253, 90, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 139, 253, 190, 2, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 190, 253, 70,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35,
                                 241, 225, 160, 108, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 81, 240, 253, 253, 119, 25, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 45, 186, 253, 253, 150, 27, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 16, 93, 252, 253, 187,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 249,
                                 253, 249, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 130,
                                 183, 253, 253, 207, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 39, 148,
                                 229, 253, 253, 253, 250, 182, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 114,
                                 221, 253, 253, 253, 253, 201, 78, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 66,
                                 213, 253, 253, 253, 253, 198, 81, 2, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 171,
                                 219, 253, 253, 253, 253, 195, 80, 9, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55, 172,
                                 226, 253, 253, 253, 253, 244, 133, 11, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 136, 253, 253, 253, 212, 135, 132, 16, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0]

        scoring_payload = {
            'input_data': [
                {
                    'values': [scoring_data, scoring_data]
                }
            ]
        }
        scores = self.client.deployments.score(TestWMLClientWithTensorflow.scoring_url, payload=scoring_payload)
        self.assertIsNotNone(scores)

    def test_07_delete_deployment(self):
        TestWMLClientWithTensorflow.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithTensorflow.deployment_uid)

    def test_08_delete_model(self):
        TestWMLClientWithTensorflow.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithTensorflow.model_uid)


if __name__ == '__main__':
    unittest.main()
