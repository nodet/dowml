import unittest
import os
import logging
from preparation_and_cleaning import *
from ibm_watson_machine_learning.utils import SCIKIT_LEARN_FRAMEWORK


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
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'invalid_but_big.tar.gz')

    def test_1_service_instance_details(self):
        TestWMLClientWithXGBoost.logger.info("Check client ...")
        self.assertTrue(type(self.client).__name__ == 'APIClient')

        TestWMLClientWithXGBoost.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()

        TestWMLClientWithXGBoost.logger.debug(details)
        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_2_publish_model(self):
        TestWMLClientWithXGBoost.logger.info("Publishing scikit-xgboost model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.AUTHOR_NAME: "IBM",
                       self.client.repository.ModelMetaNames.AUTHOR_EMAIL: "ibm@ibm.com",
                       self.client.repository.ModelMetaNames.NAME: "30Mb model",
                       self.client.repository.ModelMetaNames.FRAMEWORK_NAME: SCIKIT_LEARN_FRAMEWORK,
                       self.client.repository.ModelMetaNames.FRAMEWORK_VERSION: "0.17",
                       }
        published_model = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestWMLClientWithXGBoost.model_uid = self.client.repository.get_model_uid(published_model)
        TestWMLClientWithXGBoost.logger.info("Published model ID:" + str(TestWMLClientWithXGBoost.model_uid))
        self.assertIsNotNone(TestWMLClientWithXGBoost.model_uid)

    def test_3_publish_model_details(self):
        TestWMLClientWithXGBoost.logger.info("Get published model details ...")
        details = self.client.repository.get_details(self.model_uid)

        TestWMLClientWithXGBoost.logger.debug("Model details: " + str(details))
        self.assertTrue("30Mb model" in str(details))

    def test_4_download_model(self):
        downloaded_model_path = os.path.join('artifacts', 'downloaded_very_big_{}.tar.gz'.format(environment.lower()))

        try:
            os.remove(downloaded_model_path)
        except:
            pass

        self.client.repository.download(self.model_uid, downloaded_model_path)

        import filecmp
        self.assertTrue(filecmp.cmp(self.model_path, downloaded_model_path))

        os.remove(downloaded_model_path)

    def test_5_delete_model(self):
        TestWMLClientWithXGBoost.logger.info("Delete model ...")
        self.client.repository.delete(TestWMLClientWithXGBoost.model_uid)

if __name__ == '__main__':
    unittest.main()
