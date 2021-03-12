import unittest
from preparation_and_cleaning import *
from models_preparation import *
import logging


class TestRuntimeSpec(unittest.TestCase):
    runtime_uid = None
    runtime_url = None
    model_uid = None
    library_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestRuntimeSpec.logger.info("Service Instance: setting up credentials")

        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.model_path = os.path.join(os.getcwd(), 'artifacts', 'scikit_xgboost_model.tar.gz')
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
        lib_meta = {
            self.client.runtimes.LibraryMetaNames.NAME: "libraries_customCheck56",
            self.client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
            self.client.runtimes.LibraryMetaNames.FILEPATH: self.custom_library_path,
            self.client.runtimes.LibraryMetaNames.VERSION: "1.0",
            self.client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]}
        }
        custom_library_details = self.client.runtimes.store_library(lib_meta)
        TestRuntimeSpec.custom_library_uid = self.client.runtimes.get_library_uid(custom_library_details)

        meta = {
            self.client.runtimes.ConfigurationMetaNames.NAME: "runtime_spec_python_3.5_1",
            self.client.runtimes.ConfigurationMetaNames.DESCRIPTION: "test",
            self.client.runtimes.ConfigurationMetaNames.CONFIGURATION_FILEPATH: "/artifacts/",
            self.client.runtimes.ConfigurationMetaNames.PLATFORM: {
                "name": "python",
                "version": "3.5"
            },
            self.client.runtimes.ConfigurationMetaNames.LIBRARIES_UIDS: [TestRuntimeSpec.custom_library_uid]
        }
        runtime_details = self.client.runtimes.store(meta)
        TestRuntimeSpec.runtime_uid = self.client.runtimes.get_uid(runtime_details)
        TestRuntimeSpec.runtime_url = self.client.runtimes.get_href(runtime_details)

        self.assertTrue(TestRuntimeSpec.runtime_uid is not None)
    def test_04_download_runtime_content(self):
        try:
            os.remove('runtime.yaml')
        except:
            pass
        self.client.repository.download(TestRuntimeSpec.runtime_uid, filename='runtime.yaml')
        try:
            os.remove('runtime.yaml')
        except:
            pass

    def test_04_download_library_content(self):
        try:
            os.remove('library.tar.gz')
        except:
            pass
        self.client.repository.download(TestRuntimeSpec.library_uid, filename='library.tar.gz')
        try:
            os.remove('library.tar.gz')
        except:
            pass

    def test_3_publish_model(self):
        self.logger.info("Publishing scikit-learn model ...")

        self.client.repository.ModelMetaNames.show()

        model_props = {self.client.repository.ModelMetaNames.RUNTIME_UID: TestRuntimeSpec.runtime_uid,
                       self.client.repository.ModelMetaNames.NAME: "LOCALLY created Digits prediction model",
                       self.client.repository.ModelMetaNames.TYPE: "scikit-learn_0.19"
                       }
        published_model_details = self.client.repository.store_model(model=self.model_path, meta_props=model_props)
        TestRuntimeSpec.model_uid = self.client.repository.get_model_uid(published_model_details)
        TestRuntimeSpec.model_url = self.client.repository.get_model_href(published_model_details)
        self.logger.info("Published model ID:" + str(TestRuntimeSpec.model_uid))
        self.logger.info("Published model URL:" + str(TestRuntimeSpec.model_url))
        self.assertIsNotNone(TestRuntimeSpec.model_uid)
        self.assertIsNotNone(TestRuntimeSpec.model_url)
    def test_4_get_details(self):
        details = self.client.repository.get_details(TestRuntimeSpec.model_uid)
        runtime_url = self.client.runtimes.get_href(details)
        self.assertTrue(runtime_url == TestRuntimeSpec.runtime_url)

    def test_5_delete_model(self):
        self.client.repository.delete(TestRuntimeSpec.model_uid)

    def test_6_delete_runtime(self):

        self.client.runtimes.delete(TestRuntimeSpec.runtime_uid, with_libraries=True)


if __name__ == '__main__':
    unittest.main()
