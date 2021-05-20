import unittest
import os
from models_preparation import *
from ibm_watson_machine_learning.tests.utils import get_wml_credentials, get_cos_credentials, get_space_id
from ibm_watson_machine_learning.tests.utils.cleanup import space_cleanup
from ibm_watson_machine_learning import APIClient
import logging


class TestWMLClientWithXGBoost(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    space_name = 'tests_sdk_space'
    deployment_name = "Test deployment"
    cos_resource_instance_id = None
    model_path = None
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
        TestWMLClientWithXGBoost.space_id = get_space_id(self.wml_client, self.space_name,
                                               cos_resource_instance_id=self.cos_resource_instance_id)

        # if self.wml_client.ICP:
        #     self.wml_client.set.default_project(self.project_id)
        # else:
        self.wml_client.set.default_space(self.space_id)

    def test_2_publish_model(self):
        TestWMLClientWithXGBoost.logger.info("Creating xgboost model ...")

        model_data = create_xgboost_model_data()
        TestWMLClientWithXGBoost.model = model_data['model']

        print()
        print(model_data['params'])
        predicted = model_data['prediction']

        TestWMLClientWithXGBoost.logger.info("Predicted values:")
        TestWMLClientWithXGBoost.logger.debug(predicted)
        self.assertIsNotNone(predicted)

        try:
            model_data['model'].save_model('xgboost_model.json')

            TestWMLClientWithXGBoost.logger.info("Publishing xgboost model ...")

            self.wml_client.repository.ModelMetaNames.show()

            self.wml_client.software_specifications.list()

            sw_spec_id = self.wml_client.software_specifications.get_id_by_name('default_py3.7_opence')

            model_props = {self.wml_client.repository.ModelMetaNames.NAME: "LOCALLY created agaricus prediction model",
                           self.wml_client.repository.ModelMetaNames.TYPE: "xgboost_1.3",
                           self.wml_client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sw_spec_id
                           }
            published_model = self.wml_client.repository.store_model(model='xgboost_model.json', meta_props=model_props)
            TestWMLClientWithXGBoost.model_uid = self.wml_client.repository.get_model_uid(published_model)
            TestWMLClientWithXGBoost.logger.info("Published model ID:" + str(TestWMLClientWithXGBoost.model_uid))
            self.assertIsNotNone(TestWMLClientWithXGBoost.model_uid)
        finally:
            os.remove('xgboost_model.json')

    def test_3a_publish_model_details(self):
        TestWMLClientWithXGBoost.logger.info("Get published model details ...")
        details = self.wml_client.repository.get_details(self.model_uid)

        TestWMLClientWithXGBoost.logger.debug("Model details: " + str(details))
        self.assertTrue("LOCALLY created agaricus prediction model" in str(details))

    def test_3b_download_model(self):
        downloaded_model_path = self.wml_client.repository.download(self.model_uid, filename='downloaded_artifact.json')

        try:
            self.assertTrue(downloaded_model_path.endswith('.json'))
            with open(downloaded_model_path, 'r') as file:
                content = file.read()
                import json
                json.loads(content)
        finally:
            os.remove(downloaded_model_path)

    def test_4_create_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Create deployment ...")
        deployment = self.wml_client.deployments.create(self.model_uid, meta_props={self.wml_client.deployments.ConfigurationMetaNames.NAME:"Test deployment",self.wml_client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithXGBoost.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithXGBoost.logger.debug("Online deployment: " + str(deployment))
        TestWMLClientWithXGBoost.scoring_url = self.wml_client.deployments.get_scoring_href(deployment)
        TestWMLClientWithXGBoost.deployment_uid = self.wml_client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_5_get_deployment_details(self):
        TestWMLClientWithXGBoost.logger.info("Get deployment details ...")
        deployment_details = self.wml_client.deployments.get_details()

        TestWMLClientWithXGBoost.logger.debug("Online deployment: " + str(deployment_details))
        self.assertTrue("Test deployment" in str(deployment_details))

    def test_6_score(self):
        TestWMLClientWithXGBoost.logger.info("Online model scoring ...")

        import xgboost as xgb
        import scipy

        labels = []
        row = []
        col = []
        dat = []
        i = 0
        for l in open(os.path.join('svt', 'artifacts', 'agaricus.txt.test')):
            arr = l.split()
            labels.append(int(arr[0]))
            for it in arr[1:]:
                k, v = it.split(':')
                row.append(i)
                col.append(int(k))
                dat.append(float(v))
            i += 1
        csr = scipy.sparse.csr_matrix((dat, (row, col)))

        inp_matrix = xgb.DMatrix(csr)

        #print(TestWMLClientWithXGBoost.model.predict(inp_matrix))

        scoring_data = {
            self.wml_client.deployments.ScoringMetaNames.INPUT_DATA :[
                {
            'values': csr.getrow(0).toarray().tolist()
                }
            ]
        }

        predictions = self.wml_client.deployments.score(TestWMLClientWithXGBoost.deployment_uid, scoring_data)
        print(predictions)
        TestWMLClientWithXGBoost.logger.debug("Prediction: " + str(predictions))
        self.assertTrue(predictions is not None)

    def test_7_delete_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Delete model deployment ...")
        self.wml_client.deployments.delete(TestWMLClientWithXGBoost.deployment_uid)

    def test_8_delete_model(self):
        TestWMLClientWithXGBoost.logger.info("Delete model ...")
        self.wml_client.repository.delete(TestWMLClientWithXGBoost.model_uid)

if __name__ == '__main__':
    unittest.main()
