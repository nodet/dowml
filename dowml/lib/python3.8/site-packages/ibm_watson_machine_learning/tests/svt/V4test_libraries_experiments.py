import unittest
from preparation_and_cleaning import *
import logging


# from preparation_and_cleaning import *


class TestWMLClientWithLibrariesExperiment(unittest.TestCase):
    library_uid = None
    library_url = None
    experiment_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithLibrariesExperiment.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        print("creds " + str(self.wml_credentials))
        # reload(site)
        self.client = get_client()
        # self.cos_resource = get_cos_resource()
        # self.bucket_names = prepare_cos(self.cos_resource)

    # @classmethod
    # def tearDownClass(self):
    #     clean_cos(self.cos_resource, self.bucket_names)


    def test_01_service_instance_details(self):
        TestWMLClientWithLibrariesExperiment.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithLibrariesExperiment.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithLibrariesExperiment.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)


    def test_02_save_library(self):
        self.custom_library_path = os.path.join(os.getcwd(), 'artifacts', 'scikit_xgboost_model.tar.gz')
        lib_meta = {

            self.client.runtimes.LibraryMetaNames.NAME: "V4library_new_1",
            self.client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
            self.client.runtimes.LibraryMetaNames.VERSION: "1.0",
            self.client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.5"]},
            self.client.runtimes.LibraryMetaNames.COMMAND: "command",
            self.client.runtimes.LibraryMetaNames.FILEPATH: self.custom_library_path,

        }

        custom_library_details = self.client.runtimes.store_library(lib_meta)
        TestWMLClientWithLibrariesExperiment.library_uid = self.client.runtimes.get_library_uid(custom_library_details)
        TestWMLClientWithLibrariesExperiment.library_url = self.client.runtimes.get_library_href(custom_library_details)
        TestWMLClientWithLibrariesExperiment.logger.info("Saved library uid: " + str(TestWMLClientWithLibrariesExperiment.library_uid))

    def test_03_update_library(self):
        metadata = {
            self.client.runtimes.LibraryMetaNames.NAME: "my_library_1",
            self.client.runtimes.LibraryMetaNames.DESCRIPTION: "best_library",
        }

        library_details = self.client.runtimes.update_library(TestWMLClientWithLibrariesExperiment.library_uid, metadata)
        self.assertTrue('my_library' in str(library_details))
        TestWMLClientWithLibrariesExperiment.logger.info(library_details)
        self.assertTrue('V4library1' not in str(library_details))

    def test_04_get_library_details(self):
        details = self.client.runtimes.get_library_details()
        self.assertTrue(TestWMLClientWithLibrariesExperiment.library_uid in str(details))

        details2 = self.client.runtimes.get_library_details(TestWMLClientWithLibrariesExperiment.library_uid)
        self.assertTrue(TestWMLClientWithLibrariesExperiment.library_uid in str(details2))

    def test_05_get_library_content(self):
        try:
            os.remove('V4library_new_1-1.0.zip')
        except:
            pass

        filename = self.client.runtimes.download_library(TestWMLClientWithLibrariesExperiment.library_uid)
        self.assertTrue(filename != None)

        os.remove(filename)

        try:
            os.remove('V4library_new_1-1.0.zip')
        except:
            pass



    def test_06_save_experiment(self):
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "xxx",
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [{
                        "training_lib": {
                            "href": TestWMLClientWithLibrariesExperiment.library_url
                        }
                    }

                    ]
                }


        experiment_details = self.client.repository.store_experiment(meta_props=metadata)

        TestWMLClientWithLibrariesExperiment.experiment_uid = self.client.experiments.get_uid(experiment_details)


        experiment_specific_details = self.client.experiments.get_details(TestWMLClientWithLibrariesExperiment.experiment_uid)
        print(experiment_specific_details)
        self.assertTrue(TestWMLClientWithLibrariesExperiment.experiment_uid in str(experiment_specific_details))
        self.assertIsNotNone(TestWMLClientWithLibrariesExperiment.experiment_uid)


    def test_07_delete_library_experiment(self):
        self.client.runtimes.delete_library(TestWMLClientWithLibrariesExperiment.library_uid)
        self.client.repository.delete(TestWMLClientWithLibrariesExperiment.experiment_uid)