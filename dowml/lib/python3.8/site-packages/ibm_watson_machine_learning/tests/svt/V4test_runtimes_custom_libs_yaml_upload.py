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
        self.configuration_filepath = os.path.join(os.getcwd(), 'artifacts', 'conda_env.yaml')

    def test_1_service_instance_details(self):
        TestRuntimeSpec.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestRuntimeSpec.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()

        TestRuntimeSpec.logger.debug(details)
        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_2_create_runtime(self):
        lib_meta = {
            self.client.runtimes.LibraryMetaNames.NAME: "libraries_custom_V4_t1",
            self.client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
            self.client.runtimes.LibraryMetaNames.FILEPATH: self.custom_library_path,
            self.client.runtimes.LibraryMetaNames.VERSION: "1.0",
            self.client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]}
        }
        custom_library_details = self.client.runtimes.store_library(lib_meta)
        custom_library_uid = self.client.runtimes.get_library_uid(custom_library_details)

        meta = {
            self.client.runtimes.ConfigurationMetaNames.NAME: "runtime_spec_python_3.5_7",
            self.client.runtimes.ConfigurationMetaNames.DESCRIPTION: "test",
            self.client.runtimes.ConfigurationMetaNames.PLATFORM: {
                "name": "python",
                "version": "3.5"
            },
            self.client.runtimes.ConfigurationMetaNames.LIBRARIES_UIDS: [custom_library_uid],
            self.client.runtimes.ConfigurationMetaNames.CONFIGURATION_FILEPATH: TestRuntimeSpec.configuration_filepath
        }
        runtime_details = self.client.runtimes.store(meta)
        TestRuntimeSpec.runtime_uid = self.client.runtimes.get_uid(runtime_details)
        runtime_url = self.client.runtimes.get_href(runtime_details)

        self.assertTrue(TestRuntimeSpec.runtime_uid is not None)

    def test_3_download_yaml(self):
        try:
            os.remove('test.yaml')
        except:
            pass

        filename = self.client.runtimes.download_configuration(TestRuntimeSpec.runtime_uid, 'test.yaml')
        print(filename)
        self.assertTrue(filename == 'test.yaml')

        os.remove('test.yaml')

    def test_4_get_details(self):
        print(self.client.runtimes.get_details(TestRuntimeSpec.runtime_uid))

        print(self.client.runtimes.get_details())

    def test_5_list(self):
        self.client.runtimes.list()

    def test_6_list_custom_libs(self):
        self.client.runtimes.list_libraries(TestRuntimeSpec.runtime_uid)
        self.client.runtimes.list_libraries()

    def test_7_list_runtimes_for_libraries(self):
        self.client.runtimes._list_runtimes_for_libraries()

    def test_8_delete_runtime(self):
        self.client.runtimes.delete(TestRuntimeSpec.runtime_uid, with_libraries=True)


if __name__ == '__main__':
    unittest.main()
