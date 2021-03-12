import unittest
from preparation_and_cleaning import *
from models_preparation import *
import logging


class TestWMLClientWithXGBoost(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    logger = logging.getLogger(__name__)
    model = None

    @classmethod
    def setUpClass(self):
        TestWMLClientWithXGBoost.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()

    def test_1_service_instance_details(self):
        TestWMLClientWithXGBoost.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithXGBoost.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()

        TestWMLClientWithXGBoost.logger.debug(details)
        TestWMLClientWithXGBoost.client.set.default_space("29bf9f68-3e52-47a7-ba9d-d177b866e29f")
        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

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

        TestWMLClientWithXGBoost.logger.info("Publishing xgboost model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.NAME: "LOCALLY created agaricus prediction model",
                       self.client.repository.ModelMetaNames.TYPE: "xgboost_0.80",
                       self.client.repository.ModelMetaNames.RUNTIME_UID: "xgboost_0.80-py3"
                       }
        published_model = self.client.repository.store_model(model=model_data['model'], meta_props=model_props)
        TestWMLClientWithXGBoost.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithXGBoost.logger.info("Published model ID:" + str(TestWMLClientWithXGBoost.model_uid))
        self.assertIsNotNone(TestWMLClientWithXGBoost.model_uid)

    def test_3_publish_model_details(self):
        TestWMLClientWithXGBoost.logger.info("Get published model details ...")
        details = self.client.repository.get_details(self.model_uid)

        TestWMLClientWithXGBoost.logger.debug("Model details: " + str(details))
        self.assertTrue("LOCALLY created agaricus prediction model" in str(details))

    def test_4_create_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Create deployment ...")
        deployment = self.client.deployments.create(self.model_uid, meta_props={self.client.deployments.ConfigurationMetaNames.NAME: "Test deployment",self.client.deployments.ConfigurationMetaNames.ONLINE:{}})

        TestWMLClientWithXGBoost.logger.info("model_uid: " + self.model_uid)
        TestWMLClientWithXGBoost.logger.debug("Online deployment: " + str(deployment))
        TestWMLClientWithXGBoost.scoring_url = self.client.deployments.get_scoring_href(deployment)
        TestWMLClientWithXGBoost.deployment_uid = self.client.deployments.get_uid(deployment)
        self.assertTrue("online" in str(deployment))

    def test_5_get_deployment_details(self):
        TestWMLClientWithXGBoost.logger.info("Get deployment details ...")
        deployment_details = self.client.deployments.get_details()

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
        for l in open(os.path.join('artifacts', 'agaricus.txt.test')):
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

        print(TestWMLClientWithXGBoost.model.predict(inp_matrix))
        scoring_data = {
            self.client.deployments.ScoringMetaNames.SCORING_INPUT_DATA: [
                {
                    'fields': TestWMLClientWithXGBoost.model.feature_names,
                    'values': csr.getrow(0).toarray().tolist()
                }
            ]
        }

        predictions = self.client.deployments.score(TestWMLClientWithXGBoost.deployment_uid, scoring_data)
        TestWMLClientWithXGBoost.logger.debug("Prediction: " + str(predictions))
        self.assertTrue(predictions is not None)

    def test_7_delete_deployment(self):
        TestWMLClientWithXGBoost.logger.info("Delete model deployment ...")
        self.client.deployments.delete(TestWMLClientWithXGBoost.deployment_uid)

    def test_8_delete_model(self):
        TestWMLClientWithXGBoost.logger.info("Delete model ...")
        self.client.repository.delete(TestWMLClientWithXGBoost.model_uid)


if __name__ == '__main__':
    unittest.main()
