import unittest
import time
import io
import sys
from multiprocessing import Process, Queue
import logging
from ibm_watson_machine_learning.experiments import Experiments
from preparation_and_cleaning import *


class TestWMLClientWithKeras(unittest.TestCase):
    deployment_uid = None
    definition_1_uid = None
    definition_2_uid = None
    definition_1_url = None
    definition_2_url = None
    experiment_uid = None
    experiment_run_uid = None
    trained_model_uid = None
    train_run_id = None
    scoring_url = None
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(self):
        TestWMLClientWithKeras.logger.info("Service Instance: setting up credentials")
        self.wml_credentials = get_wml_credentials()
        self.client = get_client()
        self.cos_resource = get_cos_resource()
        self.bucket_names = prepare_cos(self.cos_resource, data_code=BOSTON)

    @classmethod
    def tearDownClass(self):
        clean_cos(self.cos_resource, self.bucket_names)

    def test_01_service_instance_details(self):
        TestWMLClientWithKeras.logger.info("Check client ...")
        self.assertTrue(self.client.__class__.__name__ == 'APIClient')

        TestWMLClientWithKeras.logger.info("Getting instance details ...")
        details = self.client.service_instance.get_details()
        TestWMLClientWithKeras.logger.debug(details)

        self.assertTrue("published_models" in str(details))
        self.assertEqual(type(details), dict)

    def test_02_save_definition1(self):
        TestWMLClientWithKeras.logger.info("Save model definition ...")

        self.client.repository.DefinitionMetaNames.show()

        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "my_training_definition",
            self.client.repository.DefinitionMetaNames.DESCRIPTION: "my_description",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 keras_boston.py"
        }

        model_content_path = './artifacts/keras_boston.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithKeras.definition_1_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithKeras.definition_1_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithKeras.logger.info("Saved model definition url: " + str(TestWMLClientWithKeras.definition_1_url))

    def test_03_save_definition2(self):
        TestWMLClientWithKeras.logger.info("Save model definition ...")

        self.client.repository.DefinitionMetaNames.show()

        metadata = {
            self.client.repository.DefinitionMetaNames.NAME: "my_training_definition",
            self.client.repository.DefinitionMetaNames.DESCRIPTION: "my_description",
            self.client.repository.DefinitionMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.DefinitionMetaNames.AUTHOR_EMAIL: "js@js.com",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_NAME: "tensorflow",
            self.client.repository.DefinitionMetaNames.FRAMEWORK_VERSION: "1.5",
            self.client.repository.DefinitionMetaNames.RUNTIME_NAME: "python",
            self.client.repository.DefinitionMetaNames.RUNTIME_VERSION: "3.5",
            self.client.repository.DefinitionMetaNames.EXECUTION_COMMAND: "python3 keras_boston.py"
        }

        model_content_path = './artifacts/keras_boston.zip'
        definition_details = self.client.repository.store_definition(training_definition=model_content_path, meta_props=metadata)
        TestWMLClientWithKeras.definition_2_url = self.client.repository.get_definition_url(definition_details)
        TestWMLClientWithKeras.definition_2_uid = self.client.repository.get_definition_uid(definition_details)
        TestWMLClientWithKeras.logger.info("Saved model definition url: " + str(TestWMLClientWithKeras.definition_2_url))

    def test_04_get_definition_details(self):
        TestWMLClientWithKeras.logger.info("Getting definition details ...")
        details = self.client.repository.get_definition_details()
        TestWMLClientWithKeras.logger.info(details)
        self.assertTrue('my_training_definition' in str(details))
        TestWMLClientWithKeras.logger.info("List definitions")
        self.client.repository.list_definitions()

    def test_05_save_experiment(self):
        metadata = {
                    self.client.repository.ExperimentMetaNames.NAME: "xxx",
                    self.client.repository.ExperimentMetaNames.AUTHOR_EMAIL: "js@js.com",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METHOD: "binary",
                    self.client.repository.ExperimentMetaNames.EVALUATION_METRICS: [],
                    self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: {
                        "type": "s3",
                        "connection": {
                            "access_key_id": "xxx",
                            "secret_access_key": "xxx",
                            "endpoint_url": "https://xxx.ibm.com"
                        },
                        "source": {
                            "bucket": "xxx",
                            "key": "xxx"
                        }
                    },
                    self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: {
                        "type": "s3",
                        "connection": {
                            "access_key_id": "xxx",
                            "secret_access_key": "xxx",
                            "endpoint_url": "https://xxx.ibm.com"
                        },
                        "target": {
                            "bucket": "xxx",
                            "key": "xxx"
                        }
                    },
                    self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                        {
                            "name": "xxx",
                            "training_definition_url": "https://xxx.ibm.com",
                            "command": "test",
                            "compute_configuration": {
                                "name": "k80"
                            }
                        }
                    ]
                }

        print(get_cos_training_data_reference(self.bucket_names))
        print(get_cos_training_results_reference(self.bucket_names))
        experiment_details = self.client.repository.store_experiment(meta_props=metadata)

        TestWMLClientWithKeras.experiment_uid = self.client.repository.get_experiment_uid(experiment_details)

        experiment_specific_details = self.client.repository.get_experiment_details(TestWMLClientWithKeras.experiment_uid)
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(experiment_specific_details))

    def test_06_update_experiment(self):
        metadata = {
            self.client.repository.ExperimentMetaNames.NAME: "my_experiment",
            self.client.repository.ExperimentMetaNames.DESCRIPTION: "mnist best model",
            self.client.repository.ExperimentMetaNames.AUTHOR_NAME: "John Smith",
            self.client.repository.ExperimentMetaNames.EVALUATION_METHOD: "multiclass",
            self.client.repository.ExperimentMetaNames.EVALUATION_METRICS: ["accuracy"],
            self.client.repository.ExperimentMetaNames.TRAINING_DATA_REFERENCE: get_cos_training_data_reference(
                self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_RESULTS_REFERENCE: get_cos_training_results_reference(
                self.bucket_names),
            self.client.repository.ExperimentMetaNames.TRAINING_REFERENCES: [
                {
                    "name": "mnist_nn",
                    "training_definition_url": TestWMLClientWithKeras.definition_1_url,
                    "compute_configuration": {"name": "k80"}
                },
                {
                    "name": "mnist_cnn",
                    "training_definition_url": TestWMLClientWithKeras.definition_2_url,
                    "compute_configuration": {"name": "k80"}
                },
            ]
        }

        experiment_details = self.client.repository.update_experiment(TestWMLClientWithKeras.experiment_uid, metadata)
        self.assertTrue('my_experiment' in str(experiment_details))
        print(experiment_details)
        self.assertTrue('xxx' not in str(experiment_details))

    def test_07_get_experiment_details(self):
        details = self.client.repository.get_experiment_details()
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(details))

        details2 = self.client.repository.get_experiment_details(TestWMLClientWithKeras.experiment_uid)
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in str(details2))

    def test_08_run_experiment(self):
        created_experiment_run_details = self.client.experiments.run(TestWMLClientWithKeras.experiment_uid, asynchronous=True)
        self.assertIsNotNone(created_experiment_run_details)
        TestWMLClientWithKeras.experiment_run_uid = Experiments.get_run_uid(created_experiment_run_details)

    def test_09_monitor(self):
        queue = Queue()
        process = Process(target=run_monitor, args=(self.client, TestWMLClientWithKeras.experiment_run_uid, queue))
        process.daemon = True
        process.start()
        process.join(timeout=600)
        process.terminate()
        self.assertFalse(queue.empty())
        self.assertTrue('training-' in queue.get())

    def test_10_get_status(self):
        start_time = time.time()
        diff_time = start_time - start_time
        while True and diff_time < 60 * 10:
            time.sleep(3)
            status = self.client.experiments.get_status(TestWMLClientWithKeras.experiment_run_uid)
            if status['state'] == 'completed' or status['state'] == 'error' or status['state'] == 'canceled':
                break
            diff_time = time.time() - start_time
        self.assertIsNotNone(status)
        self.assertTrue(status['state'] == 'completed')

    def test_11_get_all_experiments_runs_details(self):
        details = self.client.experiments.get_details()
        self.assertIsNotNone(details)

    def test_12_get_experiment_details(self):
        details = self.client.experiments.get_details(TestWMLClientWithKeras.experiment_uid)
        self.assertIsNotNone(details)

    def test_13_get_experiment_run_details(self):
        details = self.client.experiments.get_run_details(TestWMLClientWithKeras.experiment_run_uid)
        self.assertIsNotNone(details)

        self.assertIsNotNone(self.client.experiments.get_training_runs(details))
        self.assertIsNotNone(self.client.experiments.get_training_uids(details))

    def test_14_list_experiments(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs()  # Just to see values.
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list_experiments()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.repository.list_experiments()  # Just to see values.

    def test_15_list_experiment_runs_for_experiment(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.experiments.list_runs(TestWMLClientWithKeras.experiment_uid)  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.experiments.list_runs(TestWMLClientWithKeras.experiment_uid)  # Just to see values.

    def test_16_create_deployment(self):
        TestWMLClientWithKeras.logger.info("Create deployment")
        TestWMLClientWithKeras.train_run_uid = self.client.experiments.get_details(TestWMLClientWithKeras.experiment_uid)['resources'][0]['entity']['training_statuses'][0]['training_guid']
        trained_model_details = self.client.repository.store_model(TestWMLClientWithKeras.train_run_uid, meta_props={"name": "test_keras"})
        TestWMLClientWithKeras.trained_model_uid = self.client.repository.get_model_uid(trained_model_details)
        deployment_details = self.client.deployments.create(artifact_uid=TestWMLClientWithKeras.trained_model_uid, name="Test deployment", asynchronous=False)
        TestWMLClientWithKeras.logger.debug("Deployment details: " + str(deployment_details))
        TestWMLClientWithKeras.deployment_uid = self.client.deployments.get_uid(deployment_details)
        TestWMLClientWithKeras.scoring_url = self.client.deployments.get_scoring_url(deployment_details)
        self.assertTrue('online' in str(TestWMLClientWithKeras.scoring_url))

    def test_17_scoring(self):
        TestWMLClientWithKeras.logger.info("Score model")
        import numpy

        data = numpy.load(os.path.join('datasets', 'boston', 'BOSTON_DATA', 'features_test.npy'))

        scoring_data = data.tolist()
        print(scoring_data)

        scoring_payload = {'values': scoring_data}
        scores = self.client.deployments.score(TestWMLClientWithKeras.scoring_url, scoring_payload)
        self.assertIsNotNone(scores)

    def test_18_delete_deployment(self):
        TestWMLClientWithKeras.logger.info("Delete deployment")
        self.client.deployments.delete(TestWMLClientWithKeras.deployment_uid)

    def test_19_delete_model(self):
        TestWMLClientWithKeras.logger.info("Delete model")
        self.client.repository.delete(TestWMLClientWithKeras.trained_model_uid)

    def test_20_delete_experiment_run(self):
        self.client.experiments.delete(TestWMLClientWithKeras.experiment_run_uid)

    def test_21_list(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.definition_1_uid in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithKeras.definition_2_uid in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithKeras.experiment_uid in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.

    def test_22_delete_experiment(self):
        self.client.repository.delete(TestWMLClientWithKeras.experiment_uid)

    def test_23_delete_definitions(self):
        self.client.repository.delete(TestWMLClientWithKeras.definition_1_uid)
        self.client.repository.delete(TestWMLClientWithKeras.definition_2_uid)

    def test_24_list(self):
        stdout_ = sys.stdout
        captured_output = io.StringIO()  # Create StringIO object
        sys.stdout = captured_output  # and redirect stdout.
        self.client.repository.list()  # Call function.
        sys.stdout = stdout_  # Reset redirect.
        self.assertTrue(TestWMLClientWithKeras.definition_1_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithKeras.definition_2_uid not in captured_output.getvalue())
        self.assertTrue(TestWMLClientWithKeras.experiment_uid not in captured_output.getvalue())
        self.client.repository.list()  # Just to see values.


if __name__ == '__main__':
    unittest.main()
