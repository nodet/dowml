import unittest
from preparation_and_cleaning import *
from models_preparation import *
import logging


class TestRuntimeSpec(unittest.TestCase):
    runtime_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestRuntimeSpec.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.custom_library_path = os.path.join(os.getcwd(), 'artifacts', 'scikit_xgboost_model.tar.gz') # TODO

    def test_1_service_instance_details(self):
        TestRuntimeSpec.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestRuntimeSpec.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()

        TestRuntimeSpec.logger.debug(details)
        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_2_create_runtime(self):
        meta = {
            self.client.runtimes.ConfigurationMetaNames.NAME: "runtime_spec_python_3.5_new",
            self.client.runtimes.ConfigurationMetaNames.PLATFORM: {
                "name": "python",
                "version": "3.6"
            }
        }
        runtime_details = self.client.runtimes.store(meta)
        TestRuntimeSpec.runtime_uid = self.client.runtimes.get_uid(runtime_details)
        runtime_url = self.client.runtimes.get_href(runtime_details)

        self.assertTrue(TestRuntimeSpec.runtime_uid is not None)

    def test_3_get_details(self):
        self.client.runtimes.get_details(TestRuntimeSpec.runtime_uid)

    def test_4_list(self):
        self.client.runtimes.list()

    def test_5_delete_runtime(self):
        self.client.runtimes.delete(TestRuntimeSpec.runtime_uid)


if __name__ == '__main__':
    unittest.main()
