import unittest
import sys
import io
import logging
from preparation_and_cleaning import *


class TestWMLClientWithExperiment(unittest.TestCase):
    deployment_uid = None
    model_uid = None
    scoring_url = None
    library_uid = None
    library_url = None
    pipeline_uid = None
    pipeline_url = None
    trained_model_uid = None
    experiment_uid = None
    experiment_run_uid = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithExperiment.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        # reload(site)
        self.client = get_client()
        # self.cos_resource = get_cos_resource()
        # self.bucket_names = prepare_cos(self.cos_resource)

    # @classmethod
    # def tearDownClass(self):
    #     clean_cos(self.cos_resource, self.bucket_names)

    def test_01_service_instance_details(self):
        TestWMLClientWithExperiment.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithExperiment.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithExperiment.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_library(self):
        TestWMLClientWithExperiment.logger.info("Save library ...")


        self.custom_library_path = os.path.join(os.getcwd(), 'artifacts', 'scikit_xgboost_model.tar.gz')
        lib_meta = {

            self.client.runtimes.LibraryMetaNames.NAME: "libraries_custom_mine100",
            self.client.runtimes.LibraryMetaNames.DESCRIPTION: "custom libraries for scoring",
            self.client.runtimes.LibraryMetaNames.VERSION: "1.0",
            self.client.runtimes.LibraryMetaNames.PLATFORM: {"name": "python", "versions": ["3.6"]},
            self.client.runtimes.LibraryMetaNames.COMMAND: "command",
            self.client.runtimes.LibraryMetaNames.FILEPATH: self.custom_library_path,

        }

        custom_library_details = self.client.runtimes.store_library(lib_meta)
        TestWMLClientWithExperiment.library_uid = self.client.runtimes.get_library_uid(custom_library_details)
        TestWMLClientWithExperiment.library_url = self.client.runtimes.get_library_href(custom_library_details)
        TestWMLClientWithExperiment.logger.info("Saved library uid: " + str(TestWMLClientWithExperiment.library_uid))

    def test_03_save_pipeline(self):
        TestWMLClientWithExperiment.logger.info("Save pipelines ...")
        doc= {
             "doc_type": "pipeline",
             "version": "2.0",
             "primary_pipeline": "dlaas_only",
             "pipelines": [
               {
                 "id": "dlaas_only",
                 "runtime_ref": "hybrid",
                 "nodes": [
                   {
                     "id": "training",
                     "type": "model_node",
                     "op": "dl_train",
                     "runtime_ref": "DL",
                     "inputs": [
                     ],
                     "outputs": [],
                     "parameters": {
                       "name": "tf-mnist",
                       "description": "Simple MNIST model implemented in TF",
                       "command": "python3 convolutional_network.py --trainImagesFile ${DATA_DIR}/train-images-idx3-ubyte.gz --trainLabelsFile ${DATA_DIR}/train-labels-idx1-ubyte.gz --testImagesFile ${DATA_DIR}/t10k-images-idx3-ubyte.gz --testLabelsFile ${DATA_DIR}/t10k-labels-idx1-ubyte.gz --learningRate 0.001 --trainingIters 6000",
                       "training_lib_href": TestWMLClientWithExperiment.library_url,
                       "compute_configuration_name": "k80",
                       "compute_configuration_nodes": 1,
                       "target_bucket": "wml-dev-results"
                     }
                   }
                 ]
               }
             ],
             "schemas": [
               {
                 "id": "schema1",
                 "fields": [
                   {
                     "name": "text",
                     "type": "string"
                   }
                 ]
               }
             ],
             "runtimes": [
               {
                 "id": "DL",
                 "name": "tensorflow",
                 "version": "1.13-py3.6"
                 }
             ]
            }

        metadata = {
            self.client.repository.PipelineMetaNames.NAME: "my_pipeline",
            self.client.repository.PipelineMetaNames.DOCUMENT: doc


            }


        definition_details = self.client.repository.store_pipeline( meta_props=metadata)
        TestWMLClientWithExperiment.pipeline_uid = self.client.repository.get_pipeline_uid(definition_details)
        TestWMLClientWithExperiment.pipeline_url = self.client.repository.get_pipeline_href(definition_details)
        TestWMLClientWithExperiment.logger.info("Saved pipeline uid: " + str(TestWMLClientWithExperiment.pipeline_uid))

    def test_04_list_pipelines(self):
        TestWMLClientWithExperiment.logger.info("List pipelines")
        self.client.repository.list_pipelines()

    def test_05_get_uid_url(self):
        def_details = self.client.repository.get_pipeline_details(TestWMLClientWithExperiment.pipeline_uid)
        uid = self.client.repository.get_pipeline_uid(def_details)
        url = self.client.repository.get_pipeline_href(def_details)
        self.assertIsNotNone(uid)
        self.assertIsNotNone(url)

    def test_06_get_pipeline_details(self):
        TestWMLClientWithExperiment.logger.info("Getting pipeline details ...")
        details_1 = self.client.repository.get_pipeline_details(TestWMLClientWithExperiment.pipeline_uid)
        TestWMLClientWithExperiment.logger.info(details_1)
        print(details_1)
        self.assertTrue('my_pipeline' in str(details_1))

    def test_07_save_experiment(self):
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "xxx",
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [{
                        "pipeline": {
                            "href": TestWMLClientWithExperiment.pipeline_url,
                        }
                    }

                    ]
                }


        experiment_details = self.client.repository.store_experiment(meta_props=metadata)

        TestWMLClientWithExperiment.experiment_uid = self.client.experiments.get_uid(experiment_details)
        url = self.client.experiments.get_uid(experiment_details)

        experiment_specific_details = self.client.experiments.get_details(TestWMLClientWithExperiment.experiment_uid)
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in str(experiment_specific_details))
        self.assertIsNotNone(url)

    def test_08_update_experiment(self):
        metadata = {
            self.client.repository.ExperimentMetaNames.NAME: "my_experiment",
            self.client.repository.ExperimentMetaNames.DESCRIPTION: "mnist best model",
        }

        experiment_details = self.client.repository.update_experiment(TestWMLClientWithExperiment.experiment_uid, metadata)
        self.assertTrue('my_experiment' in str(experiment_details))
        TestWMLClientWithExperiment.logger.info(experiment_details)
        print(experiment_details)
        self.assertTrue('xxx' not in str(experiment_details))

    def test_09_list_experiment(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list()# Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.experiment_uid in captured_output.getvalue())
        self.client.experiments.list()  # Just to see values.
    def test_07_save_training(self):
        metadata = {

            self.client.training.ConfigurationMetaNames.TRAINING_RESULTS_REFERENCE: {"name": "name123",
                                                                                    "connection": {
                                                                                        "endpoint_url": "https://s3-api.us-geo.objectstorage.softlayer.net",
                                                                                        "access_key_id": "zfho4HT7pUIStZvSkDsl",
                                                                                        "secret_access_key": "21q66Vvxkhr4uPDacTf8F9fnzMjSUIzsZRtxrYbx"
                                                                                    },
                                                                                    "location": {
                                                                                        "bucket": "fvt-training-results"
                                                                                    },
                                                                                    "type": "s3"
                                                                                    },

            self.client.training.ConfigurationMetaNames.TRAINING_DATA_REFERENCES:
                [{
                    "name": "training_input_data",
                    "type": "s3",
                    "connection": {
                        "endpoint_url": "https://s3-api.us-geo.objectstorage.softlayer.net",
                        "access_key_id": "zfho4HT7pUIStZvSkDsl",
                        "secret_access_key": "21q66Vvxkhr4uPDacTf8F9fnzMjSUIzsZRtxrYbx"
                    },
                    "location": {
                        "bucket": "wml-dev"
                    },
                    "schema": {
                        "id": "id123_schema",
                        "fields": [
                            {
                                "name": "text",
                                "type": "string"
                            }
                        ]
                    }
                }]
            ,
            self.client.training.ConfigurationMetaNames.EXPERIMENT_UID: TestWMLClientWithExperiment.experiment_uid

        }


        training_details = self.client.training.run(meta_props=metadata)

        TestWMLClientWithExperiment.trained_model_uid = self.client.training.get_uid(training_details)
        url = self.client.training.get_uid(training_details)

        training_details = self.client.training.get_details(TestWMLClientWithExperiment.trained_model_uid)
        self.assertTrue(TestWMLClientWithExperiment.trained_model_uid in str(training_details))
        self.assertIsNotNone(url)

    def test_09_list_trainings(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.training.list()# Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithExperiment.trained_model_uid in captured_output.getvalue())
        self.client.training.list()  # Just to see values.

    def test_10_delete_training(self):
        self.client.training.delete(TestWMLClientWithExperiment.trained_model_uid)


    def test_10_delete_experiment(self):
        self.client.repository.delete(TestWMLClientWithExperiment.experiment_uid)

    def test_11_delete_pipeline_library(self):
        self.client.repository.delete(TestWMLClientWithExperiment.pipeline_uid)
        self.client.repository.delete(TestWMLClientWithExperiment.library_uid)


if __name__ == '__main__':
    unittest.main()
